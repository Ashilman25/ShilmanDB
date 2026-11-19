#include "executor/parallel/worker_pool.hpp"
#include "common/exception.hpp"
#include <string>

namespace shilmandb::parallel {

// A thread that fails to start stops the ones already running.
WorkerPool::WorkerPool(unsigned threads) {
    const unsigned background = threads > 1 ? threads - 1 : 0;
    threads_.reserve(background);
    try {
        for (unsigned i = 0; i < background; ++i) threads_.emplace_back([this, i] { Loop(i + 1); });
    } catch (...) {
        StopAll();
        throw;
    }
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
    for (std::thread& t : threads_) {
        if (t.joinable()) t.join();
    }
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
