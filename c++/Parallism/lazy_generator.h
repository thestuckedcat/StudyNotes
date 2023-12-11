#include<coroutine>
#include <memory>
#include <iostream>

template<typename T>
struct Generator {

    // �ڲ����� promise_type�����ڿ���Э�̵���Ϊ��
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;
    handle_type coro;
    //����һ��Э�̾��
    Generator(handle_type h) : coro(h) {}                         // (3)
    

    ~Generator() {
        if (coro) coro.destroy();
    }
    Generator(const Generator&) = delete;
    Generator& operator = (const Generator&) = delete;
    Generator(Generator&& oth) noexcept : coro(oth.coro) {
        oth.coro = nullptr;
    }
    Generator& operator = (Generator&& oth) noexcept {
        coro = oth.coro;
        oth.coro = nullptr;
        return *this;
    }
    T getValue() {
        return coro.promise().current_value;
    }
    bool next() {                                                // (5)
        coro.resume();
        return not coro.done();
    }
    struct promise_type {
        promise_type() = default;                               // (1)

        ~promise_type() = default;

        auto initial_suspend() {                                 // (4)
            return std::suspend_always();
        }
        auto  final_suspend() noexcept {
            return std::suspend_always();
        }
        auto get_return_object() {                               // (2)
            return Generator{ std::handle_type::from_promise(*this) };
        }
        void return_void() {
            std::suspend_never();
        }

        auto yield_value(const T value) {                        // (6) 
            current_value = value;
            return std::suspend_always{};
        }
        void unhandled_exception() {
            std::exit(1);
        }
        T current_value;
    };

};


Generator<int> getNext(int start = 0, int step = 1) noexcept {
    auto value = start;
    for (int i = 0;; ++i) {
        co_yield value;
        value += step;
    }
}

void run() {

    std::cout << std::endl;

    std::cout << "getNext():";
    auto gen = getNext();
    for (int i = 0; i <= 10; ++i) {
        gen.next();
        std::cout << " " << gen.getValue();                      // (7)
    }

    std::cout << "\n\n";

    std::cout << "getNext(100, -10):";
    auto gen2 = getNext(100, -10);
    for (int i = 0; i <= 20; ++i) {
        gen2.next();
        std::cout << " " << gen2.getValue();
    }

    std::cout << std::endl;

}