#ifndef THREAD_TEST_SAFE_QUEUE_H
#define THREAD_TEST_SAFE_QUEUE_H

#include <algorithm>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <optional>
#include <queue>
#include <vector>
#include <thread>

class non_empty_queue : public std::exception {
    std::string what_;
public:
    explicit non_empty_queue(std::string msg) { what_ = std::move(msg); }
    [[nodiscard]] const char* what() const noexcept override  { return what_.c_str(); }
};

template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    // Moved out of public interface to prevent races between this
    // and pop().
    [[nodiscard]] bool empty() const {
        return queue_.empty();
    }

public:
    ThreadSafeQueue() = default;
    ThreadSafeQueue(const ThreadSafeQueue<T> &) = delete ;
    ThreadSafeQueue& operator=(const ThreadSafeQueue<T> &) = delete ;

    ThreadSafeQueue(ThreadSafeQueue<T>&& other) noexcept(false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!empty()) {
            throw non_empty_queue("Moving into a non-empty queue");
        }
        queue_ = std::move(other.queue_);
    }

    virtual ~ThreadSafeQueue() noexcept(false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!empty()) {
            throw non_empty_queue("Destroying a non-empty queue");
        }
    }

    [[nodiscard]] unsigned long size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]{return !empty();});

        T tmp = queue_.front();
        queue_.pop();

        lock.unlock();
        cv_.notify_all();

        return tmp;
    }

    void push(const T &item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(item);
        }
        cv_.notify_all();
    }

    void push_list(const std::vector<T> &item_list) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& item: item_list)
                queue_.push(item);
        }
        cv_.notify_all();
    }
};


template<typename T>
class ThreadSafeValue {
public:
    ThreadSafeValue() = default;

    // Multiple threads/readers can read the counter's value at the same time.
    T get() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return value_;
    }

    // Only one thread/writer can increment/write the counter's value.
    void set(const T& value) {
        // You can also use lock_guard here.
        std::unique_lock<std::shared_mutex> lock(mutex_);
        value_ = value;
    }

private:
    mutable std::shared_mutex mutex_;
    T value_;
};

#endif //THREAD_TEST_SAFE_QUEUE_H
