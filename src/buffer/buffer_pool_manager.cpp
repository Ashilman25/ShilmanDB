#include "buffer/buffer_pool_manager.hpp"

namespace shilmandb {

BufferPoolManager::BufferPoolManager(size_t pool_size, DiskManager* disk_manager, std::unique_ptr<EvictionPolicy> eviction_policy) : pool_size_(pool_size), pages_(new Page[pool_size]), disk_manager_(disk_manager), eviction_policy_(std::move(eviction_policy)) {
    for (size_t i = 0; i < pool_size_; ++i) {
        free_list_.push_back(static_cast<frame_id_t>(i));
    }
}

BufferPoolManager::~BufferPoolManager() {
    try {
        FlushAllPages();
    } catch (...) {
        //log don't throw from destructor
    }
    delete[] pages_;
}

Page* BufferPoolManager::FetchPage(page_id_t page_id) {
    // Cache hit
    if (auto it = page_table_.find(page_id); it != page_table_.end()) {
        auto frame_id = it->second;
        pages_[frame_id].Pin();
        eviction_policy_->RecordAccess(frame_id);
        eviction_policy_->SetEvictable(frame_id, false);
        ++hit_count_;
        return &pages_[frame_id];
    }

    // Cache miss, find a frame
    frame_id_t frame_id;
    if (!FindVictimFrame(&frame_id)) {
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

    // Load requested page
    frame.ResetMemory();
    disk_manager_->ReadPage(page_id, frame.GetData());
    frame.SetPageId(page_id);
    frame.Pin();

    page_table_[page_id] = frame_id;
    eviction_policy_->RecordAccess(frame_id);
    eviction_policy_->SetEvictable(frame_id, false);
    ++miss_count_;
    return &frame;
}

Page* BufferPoolManager::NewPage(page_id_t* page_id) {
    frame_id_t frame_id;
    if (!FindVictimFrame(&frame_id)) {
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

    page_table_[new_page_id] = frame_id;
    eviction_policy_->RecordAccess(frame_id);
    eviction_policy_->SetEvictable(frame_id, false);

    *page_id = new_page_id;
    return &frame;
}

bool BufferPoolManager::UnpinPage(page_id_t page_id, bool is_dirty) {
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
        eviction_policy_->SetEvictable(it->second, true);
    }
    return true;
}

bool BufferPoolManager::FlushPage(page_id_t page_id) {
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
    for (size_t i = 0; i < pool_size_; ++i) {
        if (pages_[i].GetPageId() != INVALID_PAGE_ID && pages_[i].IsDirty()) {
            disk_manager_->WritePage(pages_[i].GetPageId(), pages_[i].GetData());
            pages_[i].ClearDirty();
        }
    }
}

bool BufferPoolManager::DeletePage(page_id_t page_id) {
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
    eviction_policy_->Remove(frame_id);
    frame.ResetMemory();
    free_list_.push_back(frame_id);
    return true;
}

bool BufferPoolManager::FindVictimFrame(frame_id_t* frame_id) {
    if (!free_list_.empty()) {
        *frame_id = free_list_.front();
        free_list_.pop_front();
        return true;
    }

    auto victim = eviction_policy_->Evict();
    if (victim.has_value()) {
        *frame_id = *victim;
        return true;
    }
    return false;
}

uint64_t BufferPoolManager::GetHitCount()  const { return hit_count_; }
uint64_t BufferPoolManager::GetMissCount() const { return miss_count_; }
void BufferPoolManager::ResetStats() { hit_count_ = 0; miss_count_ = 0; }
size_t BufferPoolManager::GetPoolSize() const { return pool_size_; }

}  // namespace shilmandb
