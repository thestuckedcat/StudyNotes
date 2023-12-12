#include<coroutine>
#include <memory>
#include <iostream>


namespace lazy_generator {
    template<typename T>
    struct Generator {

        // 内部定义 promise_type，用于控制协程的行为。
        struct promise_type;
        using handle_type = std::coroutine_handle<promise_type>;
        handle_type coro;

    public:
        //接收一个协程句柄
        Generator(handle_type h) : coro(h) {}                         // (3)


        ~Generator() {
            if (coro) coro.destroy();
            //隐式转换操作符，用于检查协程句柄是否非空（即是否指向一个有效的协程）。
        }

        Generator(const Generator&) = delete;
        Generator& operator = (const Generator&) = delete;


        Generator(Generator&& oth) noexcept : coro(oth.coro) {
            oth.coro = nullptr;//移动构造
        }
        Generator& operator = (Generator&& oth) noexcept {
            coro = oth.coro;
            oth.coro = nullptr;
        }





        T getValue() {
            return coro.promise().current_value;

        }
        bool next() {                                                // 恢复线程
            coro.resume();
            return not coro.done();
        }



        struct promise_type {
            promise_type() = default;                               // (1)

            ~promise_type() = default;

            //固有组件
            std::suspend_always initial_suspend() {                                 // (4)
                return std::suspend_always();
            }
            std::suspend_always  final_suspend() noexcept {
                return std::suspend_always();
            }
            Generator<T>  get_return_object() {                               // (2)
                return Generator<T>{ std::coroutine_handle<promise_type>::from_promise(*this) };
            }

            //co_return组件，此例中co_routine结束默认调用
            void return_value(const T& value) {

            }

            // co_yield组件
            std::suspend_always yield_value(const T value) {                        // (6) 
                current_value = value;
                return std::suspend_always{};
            }

            //固定组件
            void unhandled_exception() {
                std::exit(1);
            }

            //自定义组件
            T current_value;
        };

    };

    //使用yield value，这里的Generator并不是做返回值作用，而是相当于一个包装
    Generator<int> getNext(int start = 0, int step = 1) noexcept {
        auto value = start;
        for (int i = 0;; ++i) {
            co_yield value;//语句将当前的 value 传递给调用者（promise内），并暂时挂起协程。当协程被恢复时，执行从 co_yield 之后的语句继续。
            value += step;
        }
    }

    void run() {

        std::cout << std::endl;

        std::cout << "getNext():";
        auto gen = getNext();
        for (int i = 0; i <= 10; ++i) {
            gen.next();//恢复线程
            //在getNext()中，计算完后自动挂起线程，将值传给promise的current_value
            std::cout << " " << gen.getValue();    //读取current_value
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

}