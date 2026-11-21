#include "executor/parallel/worker_pool.hpp"
#include "common/exception.hpp"
#include <sys/resource.h>
#include <unistd.h>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <memory>
#include <string>
#include <system_error>

namespace shilmandb::parallel {

namespace {

constexpr size_t kUnlimitedStack = size_t{8} << 20;  // used when the stack limit is unlimited

// The main thread's stack size: the soft limit, or 8 MiB when unlimited.
// At least PTHREAD_STACK_MIN, rounded up to whole pages.
size_t WorkerStackSize() {
    size_t size = kUnlimitedStack;
    rlimit limit{};
    if (getrlimit(RLIMIT_STACK, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY) size = static_cast<size_t>(limit.rlim_cur);
    size = std::max(size, static_cast<size_t>(PTHREAD_STACK_MIN));
    const long page_size = sysconf(_SC_PAGESIZE);
    const size_t page = page_size > 0 ? static_cast<size_t>(page_size) : 4096;
    return (size + page - 1) / page * page;
}

// Throws on a pthread error code, as std::thread does.
void Check(int rc, const char* what) {
    if (rc != 0) throw std::system_error(rc, std::generic_category(), what);
}

// Thread attributes asking for the worker stack size.
class StackAttr {
public:
    StackAttr() {
        Check(pthread_attr_init(&attr_), "pthread_attr_init");
        const int rc = pthread_attr_setstacksize(&attr_, WorkerStackSize());
        if (rc != 0) pthread_attr_destroy(&attr_);
        Check(rc, "pthread_attr_setstacksize");
    }
    ~StackAttr() { pthread_attr_destroy(&attr_); }
    StackAttr(const StackAttr&) = delete;
    StackAttr& operator=(const StackAttr&) = delete;

    [[nodiscard]] const pthread_attr_t* Get() const { return &attr_; }

private:
    pthread_attr_t attr_{};
};

// What a background thread needs to run its loop.
struct ThreadStart {
    WorkerPool* pool;
    unsigned index;
};

}  // namespace

// A thread that fails to start stops the ones already running.
WorkerPool::WorkerPool(unsigned threads) {
    const unsigned background = threads > 1 ? threads - 1 : 0;
    if (background == 0) return;
    threads_.reserve(background);
    try {
        const StackAttr attr;
        for (unsigned i = 0; i < background; ++i) {
            auto start = std::make_unique<ThreadStart>(ThreadStart{this, i + 1});
            pthread_t thread{};
            Check(pthread_create(&thread, attr.Get(), &WorkerPool::ThreadMain, start.get()), "pthread_create");
            (void)start.release();      // the thread owns it now
            threads_.push_back(thread);  // can't throw: reserved above
        }
    } catch (...) {
        StopAll();
        throw;
    }
}

// Takes over its start record, then loops until the pool stops.
void* WorkerPool::ThreadMain(void* arg) noexcept {
    const std::unique_ptr<ThreadStart> start(static_cast<ThreadStart*>(arg));
    start->pool->Loop(start->index);
    return nullptr;
}

WorkerPool::~WorkerPool() {
    StopAll();
}

// Wakes every thread with the stop flag set, then waits for each to end.
void WorkerPool::StopAll() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    start_.notify_all();
    for (pthread_t thread : threads_) pthread_join(thread, nullptr);
    threads_.clear();
}

// Background workers start under the lock, so they see everything set before Run.
void WorkerPool::Run(unsigned workers, const std::function<void(unsigned)>& fn) {
    if (workers > Size()) {
        throw DatabaseException("a run of " + std::to_string(workers) + " workers on a pool of " + std::to_string(Size()));
    }
    if (workers == 0) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = &fn;
        workers_ = workers;
        running_ = workers - 1;
        error_ = nullptr;
        ++run_;
    }
    if (workers > 1) start_.notify_all();

    Call(fn, 0);

    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [this] { return running_ == 0; });
        job_ = nullptr;
        error = error_;
        error_ = nullptr;
    }
    if (error) std::rethrow_exception(error);
}

// A run can't end until every worker in it returns, so none misses its run.
void WorkerPool::Loop(unsigned index) {
    uint64_t seen = 0;
    for (;;) {
        const std::function<void(unsigned)>* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            start_.wait(lock, [&] { return stopping_ || run_ != seen; });
            if (stopping_) return;
            seen = run_;
            if (index >= workers_) continue;
            job = job_;
        }
        Call(*job, index);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--running_ == 0) finished_.notify_one();
        }
    }
}

// The first exception wins; later ones are dropped.
void WorkerPool::Call(const std::function<void(unsigned)>& fn, unsigned index) {
    try {
        fn(index);
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) error_ = std::current_exception();
    }
}

}  // namespace shilmandb::parallel
