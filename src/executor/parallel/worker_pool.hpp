#pragma once
#include <pthread.h>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <vector>

namespace shilmandb::parallel {

// Threads that wait between runs and work on one job at a time.
// The calling thread is worker 0,
// so a pool of n threads starts n - 1.
// Each gets the main thread's stack size, so deep expressions don't crash them.
// One caller at a time: Run is not reentrant.
class WorkerPool {
public:
    // Starts threads - 1 background threads (none for 0 or 1).
    // Throws std::system_error when one can't start.
    explicit WorkerPool(unsigned threads);

    // Stops and joins every background thread.
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    // Workers a run may use, the calling thread included.
    [[nodiscard]] unsigned Size() const { return static_cast<unsigned>(threads_.size()) + 1; }

    // Runs fn(w) for w = 0..workers-1 and waits for all.
    // Worker 0 runs on the calling thread.
    // Rethrows the first exception any worker threw, of any type, once all have returned.
    // Throws DatabaseException when workers is more than Size().
    void Run(unsigned workers, const std::function<void(unsigned)>& fn);

private:
    // A background thread's entry point; arg is its start record.
    static void* ThreadMain(void* arg) noexcept;

    // A background thread's loop: wait for a run,
    // and take part if its index is in it.
    void Loop(unsigned index);

    // Calls fn(index), keeping the first exception of the run.
    void Call(const std::function<void(unsigned)>& fn, unsigned index);

    // Stops the threads started so far and joins them.
    void StopAll();

    std::vector<pthread_t> threads_;    // thread i is worker i + 1
    std::mutex mutex_;
    std::condition_variable start_;     // a new run, or stopping
    std::condition_variable finished_;  // the last background worker of a run returned
    const std::function<void(unsigned)>* job_{nullptr};
    uint64_t run_{0};                   // bumped once per run
    unsigned workers_{0};               // workers in the current run
    unsigned running_{0};               // background workers of the run not yet returned
    std::exception_ptr error_;          // the run's first exception
    bool stopping_{false};
};

}  // namespace shilmandb::parallel
