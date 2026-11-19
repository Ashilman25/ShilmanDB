#include "buffer/buffer_pool_manager.hpp"
#include "common/exception.hpp"
#include <algorithm>
#include <string>

namespace shilmandb {

BufferPoolManager::BufferPoolManager(size_t pool_size, DiskManager* disk_manager, std::unique_ptr<EvictionPolicy> eviction_policy) : pool_size_(pool_size), pages_(new Page[pool_size]), disk_manager_(disk_manager), eviction_policy_(std::move(eviction_policy)), policy_(eviction_policy_.get()), last_seen_(pool_size, 0), uses_access_types_(eviction_policy_->UsesAccessTypes()), frame_run_(pool_size, 0), owner_(std::this_thread::get_id()), loading_(pool_size, false) {
    for (size_t i = 0; i < pool_size_; ++i) {
        free_list_.push_back(static_cast<frame_id_t>(i));
    }
}

BufferPoolManager::~BufferPoolManager() {
    DisableTracing();
    try {
        FlushAllPages();
    } catch (...) {
        //log don't throw from destructor
    }
    delete[] pages_;
}

Page* BufferPoolManager::FetchPage(page_id_t page_id) {
    if (concurrent_) return FetchPageConcurrent(page_id);
    AssertOwnerThread();
    // Cache hit
    if (auto it = page_table_.find(page_id); it != page_table_.end()) {
        auto frame_id = it->second;
        pages_[frame_id].Pin();
        RecordPolicyAccess(frame_id, page_id, false);
        policy_->SetEvictable(frame_id, false);
        ++hit_count_;
        RecordVisit(page_id, frame_id, false);
        if (trace_enabled_) TraceFetch(page_id);
        return &pages_[frame_id];
    }

    // Cache miss, find a frame
    frame_id_t frame_id;
    if (!FindVictimFrame(page_id, &frame_id)) {
        return nullptr;
    }

    // Flush dirty victim
    auto& frame = pages_[frame_id];
    if (frame.GetPageId() != INVALID_PAGE_ID) {
        if (frame.IsDirty()) {
            disk_manager_->WritePage(frame.GetPageId(), frame.GetData());
        }
        page_table_.erase(frame.GetPageId());
    }

    // Load requested page. The read overwrites the whole frame, so only the metadata needs clearing.
    // If it fails, the frame goes back to the free list.
    frame.ClearMetadata();
    try {
        disk_manager_->ReadPage(page_id, frame.GetData());
    } catch (...) {
        free_list_.push_back(frame_id);
        throw;
    }
    frame.SetPageId(page_id);
    frame.Pin();

    page_table_[page_id] = frame_id;
    RecordPolicyAccess(frame_id, page_id, true);
    policy_->SetEvictable(frame_id, false);
    ++miss_count_;
    RecordVisit(page_id, frame_id, true);
    if (trace_enabled_) TraceFetch(page_id);
    return &frame;
}

Page* BufferPoolManager::NewPage(page_id_t* page_id) {
    auto lock = LatchIfConcurrent();
    AssertOwnerThread();
    frame_id_t frame_id;
    if (!FindVictimFrame(INVALID_PAGE_ID, &frame_id)) {
        return nullptr;
    }

    // Flush dirty victim
    auto& frame = pages_[frame_id];
    if (frame.GetPageId() != INVALID_PAGE_ID) {
        if (frame.IsDirty()) {
            disk_manager_->WritePage(frame.GetPageId(), frame.GetData());
        }
        page_table_.erase(frame.GetPageId());
    }

    // Allocate new page
    auto new_page_id = disk_manager_->AllocatePage();
    frame.ResetMemory();
    frame.SetPageId(new_page_id);
    frame.Pin();
    last_seen_[frame_id] = 0;  // not fetched yet

    page_table_[new_page_id] = frame_id;
    RecordPolicyAccess(frame_id, new_page_id, true);  // a new page counts as a new use
    policy_->SetEvictable(frame_id, false);

    *page_id = new_page_id;
    return &frame;
}

bool BufferPoolManager::UnpinPage(page_id_t page_id, bool is_dirty) {
    auto lock = LatchIfConcurrent();
    AssertOwnerThread();
    auto it = page_table_.find(page_id);
    if (it == page_table_.end()) {
        return false;
    }

    auto& frame = pages_[it->second];
    if (frame.GetPinCount() == 0) {
        return false;
    }

    frame.Unpin();
    if (is_dirty) {
        frame.MarkDirty();
    }
    if (frame.GetPinCount() == 0) {
        policy_->SetEvictable(it->second, true);
    }
    return true;
}

bool BufferPoolManager::FlushPage(page_id_t page_id) {
    if (concurrent_) return FlushPageConcurrent(page_id);
    AssertOwnerThread();
    auto it = page_table_.find(page_id);
    if (it == page_table_.end()) {
        return false;
    }

    auto& frame = pages_[it->second];
    disk_manager_->WritePage(page_id, frame.GetData());
    frame.ClearDirty();
    return true;
}

void BufferPoolManager::FlushAllPages() {
    auto lock = LatchIfConcurrent();
    AssertOwnerThread();
    for (size_t i = 0; i < pool_size_; ++i) {
        if (pages_[i].GetPageId() != INVALID_PAGE_ID && pages_[i].IsDirty()) {
            disk_manager_->WritePage(pages_[i].GetPageId(), pages_[i].GetData());
            pages_[i].ClearDirty();
        }
    }
}

void BufferPoolManager::DropAllPages() {
    if (concurrent_) {
        DropAllPagesConcurrent();
        return;
    }
    AssertOwnerThread();
    // Check first, so a pinned page leaves the pool exactly as it was
    for (size_t i = 0; i < pool_size_; ++i) {
        if (pages_[i].GetPinCount() > 0) {
            throw DatabaseException("Cannot drop pages: page " + std::to_string(pages_[i].GetPageId()) + " is pinned");
        }
    }

    FlushAllPages();
    for (size_t i = 0; i < pool_size_; ++i) {
        pages_[i].ClearMetadata();
    }
    page_table_.clear();

    // Hand frames out in the same order as a new pool
    free_list_.clear();
    for (size_t i = 0; i < pool_size_; ++i) {
        free_list_.push_back(static_cast<frame_id_t>(i));
    }

    policy_->Reset();
    std::fill(last_seen_.begin(), last_seen_.end(), 0);
    policy_last_page_ = INVALID_PAGE_ID;
}

bool BufferPoolManager::DeletePage(page_id_t page_id) {
    auto lock = LatchIfConcurrent();
    AssertOwnerThread();
    auto it = page_table_.find(page_id);
    if (it == page_table_.end()) {
        return false;
    }

    auto frame_id = it->second;
    auto& frame = pages_[frame_id];
    if (frame.GetPinCount() > 0) {
        return false;
    }

    page_table_.erase(it);
    policy_->Remove(frame_id);
    frame.ResetMemory();
    free_list_.push_back(frame_id);
    return true;
}

// Policies that count visits get the access type; the others get the plain call
void BufferPoolManager::RecordPolicyAccess(frame_id_t frame_id, page_id_t page_id, bool is_miss) {
    if (uses_access_types_) {
        policy_->RecordAccess(frame_id, page_id, NextAccessType(page_id, frame_id, is_miss));
    } else {
        policy_->RecordAccess(frame_id, page_id);
    }
}

// A run is a stretch of fetches to one page. A return to a resident page within
// kVisitWindow runs of its frame's last fetch is the same visit.
AccessType BufferPoolManager::NextAccessType(page_id_t page_id, frame_id_t frame_id, bool is_miss) {
    const bool page_change = page_id != policy_last_page_;
    if (page_change) {
        ++policy_run_;
        policy_last_page_ = page_id;
    }
    const uint64_t previous_run = frame_run_[frame_id];
    frame_run_[frame_id] = policy_run_;

    if (is_miss) return AccessType::NEW_USE;
    if (!page_change) return AccessType::REPEAT;
    return policy_run_ - previous_run <= kVisitWindow ? AccessType::TOUCH : AccessType::NEW_USE;
}

void BufferPoolManager::RecordVisit(page_id_t page_id, frame_id_t frame_id, bool is_miss) {
    if (page_id != last_page_id_) {
        ++page_changes_;
        last_page_id_ = page_id;
    }
    const bool new_visit = is_miss || last_seen_[frame_id] <= epoch_start_ || page_changes_ - last_seen_[frame_id] > kVisitWindow;
    if (new_visit) {
        ++visit_count_;
    }
    last_seen_[frame_id] = page_changes_;
}

bool BufferPoolManager::FindVictimFrame(page_id_t incoming_page, frame_id_t* frame_id) {
    if (!free_list_.empty()) {
        *frame_id = free_list_.front();
        free_list_.pop_front();
        return true;
    }

    auto victim = policy_->Evict(incoming_page);
    if (victim.has_value()) {
        *frame_id = *victim;
        return true;
    }
    return false;
}

uint64_t BufferPoolManager::GetHitCount()  const { auto lock = LatchIfConcurrent(); return hit_count_; }
uint64_t BufferPoolManager::GetMissCount() const { auto lock = LatchIfConcurrent(); return miss_count_; }
uint64_t BufferPoolManager::GetVisitCount() const { auto lock = LatchIfConcurrent(); return visit_count_; }

void BufferPoolManager::ResetStats() {
    auto lock = LatchIfConcurrent();
    AssertOwnerThread();
    latch_wait_ = std::chrono::nanoseconds{0};
    hit_count_ = 0;
    miss_count_ = 0;
    visit_count_ = 0;
    epoch_start_ = page_changes_;
    last_page_id_ = INVALID_PAGE_ID;  // the next fetch counts as a page change
    if (timed_policy_) {
        timed_policy_->ResetStats();
    }
}

void BufferPoolManager::SetPolicyTiming(bool on) {
    auto lock = LatchIfConcurrent();
    AssertOwnerThread();
    if (on && !timed_policy_) {
        timed_policy_ = std::make_unique<TimedEvictionPolicy>(eviction_policy_.get());
    }
    // Off goes straight back to the real policy; the counts stay readable
    policy_ = on ? static_cast<EvictionPolicy*>(timed_policy_.get()) : eviction_policy_.get();
}

PolicyStats BufferPoolManager::GetPolicyStats() const {
    auto lock = LatchIfConcurrent();
    return timed_policy_ ? timed_policy_->GetStats() : PolicyStats{};
}

size_t BufferPoolManager::GetPoolSize() const { return pool_size_; }

const Page& BufferPoolManager::GetPage(frame_id_t frame_id) const {
    auto lock = LatchIfConcurrent();
    assert(frame_id < pool_size_ && "frame_id out of bounds");
    return pages_[frame_id];
}

void BufferPoolManager::EnableTracing(const std::string& trace_path, bool collapse_repeats) {
    if (concurrent_) {
        auto lock = Latch();
        CloseTrace();
        OpenTrace(trace_path, collapse_repeats);
        return;
    }
    AssertOwnerThread();
    DisableTracing();
    trace_file_.open(trace_path, std::ios::out | std::ios::trunc);
    if (!trace_file_.is_open()) {
        throw DatabaseException("Failed to open trace file: " + trace_path);
    }
    trace_file_ << "timestamp,page_id\n";
    access_counter_ = 0;
    trace_collapse_ = collapse_repeats;
    last_traced_page_ = INVALID_PAGE_ID;
    trace_enabled_ = true;
}

void BufferPoolManager::TraceFetch(page_id_t page_id) {
    if (!trace_collapse_ || page_id != last_traced_page_) {
        trace_file_ << access_counter_ << ',' << page_id << '\n';
    }
    last_traced_page_ = page_id;
    ++access_counter_;
}

void BufferPoolManager::DisableTracing() {
    auto lock = LatchIfConcurrent();
    AssertOwnerThread();
    if (trace_enabled_) {
        trace_file_.flush();
        trace_file_.close();
        trace_enabled_ = false;
    }
}

//  Concurrent mode 

// Only the calling thread switches it, with no worker inside, so no latch
void BufferPoolManager::SetConcurrent(bool on) {
    owner_ = std::this_thread::get_id();
    concurrent_ = on;
}

std::chrono::nanoseconds BufferPoolManager::GetLatchWaitTime() const {
    auto lock = LatchIfConcurrent();
    return latch_wait_;
}

// Takes the latch; only a blocked wait is timed
std::unique_lock<std::mutex> BufferPoolManager::Latch() const {
    std::unique_lock<std::mutex> lock(latch_, std::try_to_lock);
    if (!lock.owns_lock()) {
        const auto start = std::chrono::steady_clock::now();
        lock.lock();
        latch_wait_ += std::chrono::steady_clock::now() - start;
    }
    return lock;
}

// Finds a page once no thread is reading it in. A woken waiter looks again:
// the read may have failed, or the page been evicted since.
std::unordered_map<page_id_t, frame_id_t>::iterator BufferPoolManager::WaitUntilLoaded(std::unique_lock<std::mutex>& lock, page_id_t page_id) {
    for (;;) {
        auto it = page_table_.find(page_id);
        if (it == page_table_.end() || !loading_[it->second]) return it;
        load_done_.wait(lock);
    }
}

// FetchPage under the latch, with the disk read done outside it
Page* BufferPoolManager::FetchPageConcurrent(page_id_t page_id) {
    auto lock = Latch();

    // Cache hit, once any read of the page has finished
    if (auto it = WaitUntilLoaded(lock, page_id); it != page_table_.end()) {
        auto frame_id = it->second;
        pages_[frame_id].Pin();
        RecordPolicyAccess(frame_id, page_id, false);
        policy_->SetEvictable(frame_id, false);
        ++hit_count_;
        RecordVisit(page_id, frame_id, false);
        if (trace_enabled_) TraceFetch(page_id);
        return &pages_[frame_id];
    }

    // Cache miss. The policy calls share this hold: ARC's Evict(page) is finished by the
    // RecordAccess after it, and another thread's miss in between would undo it.
    frame_id_t frame_id;
    if (!FindVictimFrame(page_id, &frame_id)) {
        return nullptr;
    }

    // Flush dirty victim
    auto& frame = pages_[frame_id];
    if (frame.GetPageId() != INVALID_PAGE_ID) {
        if (frame.IsDirty()) {
            disk_manager_->WritePage(frame.GetPageId(), frame.GetData());
        }
        page_table_.erase(frame.GetPageId());
    }

    // Enter the page as loading, pinned, so no one evicts it or reads it early
    frame.ClearMetadata();
    frame.SetPageId(page_id);
    frame.Pin();
    page_table_[page_id] = frame_id;
    loading_[frame_id] = true;
    try {
        RecordPolicyAccess(frame_id, page_id, true);
        policy_->SetEvictable(frame_id, false);
    } catch (...) {
        AbandonLoad(page_id, frame_id);
        throw;
    }

    // Read with the latch released; other fetches of this page wait for it
    lock.unlock();
    try {
        disk_manager_->ReadPage(page_id, frame.GetData());
    } catch (...) {
        lock = Latch();
        AbandonLoad(page_id, frame_id);
        throw;
    }

    // Counted only once the read worked, like the plain miss path
    lock = Latch();
    loading_[frame_id] = false;
    ++miss_count_;
    RecordVisit(page_id, frame_id, true);
    if (trace_enabled_) TraceFetch(page_id);
    load_done_.notify_all();
    return &frame;
}

// Undoes a failed load: the policy forgets the frame, and it is freed
void BufferPoolManager::AbandonLoad(page_id_t page_id, frame_id_t frame_id) {
    policy_->Remove(frame_id);
    page_table_.erase(page_id);
    pages_[frame_id].ClearMetadata();
    free_list_.push_back(frame_id);
    loading_[frame_id] = false;
    load_done_.notify_all();
}

// FlushPage under the latch, after any read of the page has finished
bool BufferPoolManager::FlushPageConcurrent(page_id_t page_id) {
    auto lock = Latch();
    auto it = WaitUntilLoaded(lock, page_id);
    if (it == page_table_.end()) {
        return false;
    }

    auto& frame = pages_[it->second];
    disk_manager_->WritePage(page_id, frame.GetData());
    frame.ClearDirty();
    return true;
}

// DropAllPages in one latch hold. A page being read in is pinned, so refuses.
void BufferPoolManager::DropAllPagesConcurrent() {
    auto lock = Latch();
    for (size_t i = 0; i < pool_size_; ++i) {
        if (pages_[i].GetPinCount() > 0) {
            throw DatabaseException("Cannot drop pages: page " + std::to_string(pages_[i].GetPageId()) + " is pinned");
        }
    }

    for (size_t i = 0; i < pool_size_; ++i) {
        if (pages_[i].GetPageId() != INVALID_PAGE_ID && pages_[i].IsDirty()) {
            disk_manager_->WritePage(pages_[i].GetPageId(), pages_[i].GetData());
            pages_[i].ClearDirty();
        }
    }
    for (size_t i = 0; i < pool_size_; ++i) {
        pages_[i].ClearMetadata();
    }
    page_table_.clear();

    free_list_.clear();
    for (size_t i = 0; i < pool_size_; ++i) {
        free_list_.push_back(static_cast<frame_id_t>(i));
    }

    policy_->Reset();
    std::fill(last_seen_.begin(), last_seen_.end(), 0);
    policy_last_page_ = INVALID_PAGE_ID;
}

// DisableTracing's work, for callers that already hold the latch
void BufferPoolManager::CloseTrace() {
    if (trace_enabled_) {
        trace_file_.flush();
        trace_file_.close();
        trace_enabled_ = false;
    }
}

// EnableTracing's work after closing the old trace, for callers that hold the latch
void BufferPoolManager::OpenTrace(const std::string& trace_path, bool collapse_repeats) {
    trace_file_.open(trace_path, std::ios::out | std::ios::trunc);
    if (!trace_file_.is_open()) {
        throw DatabaseException("Failed to open trace file: " + trace_path);
    }
    trace_file_ << "timestamp,page_id\n";
    access_counter_ = 0;
    trace_collapse_ = collapse_repeats;
    last_traced_page_ = INVALID_PAGE_ID;
    trace_enabled_ = true;
}

}  // namespace shilmandb
