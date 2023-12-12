#include<coroutine>
#include <memory>
#include <iostream>


namespace lazy_generator {
    template<typename T>
    struct Generator {

        // �ڲ����� promise_type�����ڿ���Э�̵���Ϊ��
        struct promise_type;
        using handle_type = std::coroutine_handle<promise_type>;
        handle_type coro;

    public:
        //����һ��Э�̾��
        Generator(handle_type h) : coro(h) {}                         // (3)


        ~Generator() {
            if (coro) coro.destroy();
            //��ʽת�������������ڼ��Э�̾���Ƿ�ǿգ����Ƿ�ָ��һ����Ч��Э�̣���
        }

        Generator(const Generator&) = delete;
        Generator& operator = (const Generator&) = delete;


        Generator(Generator&& oth) noexcept : coro(oth.coro) {
            oth.coro = nullptr;//�ƶ�����
        }
        Generator& operator = (Generator&& oth) noexcept {
            coro = oth.coro;
            oth.coro = nullptr;
        }





        T getValue() {
            return coro.promise().current_value;

        }
        bool next() {                                                // �ָ��߳�
            coro.resume();
            return not coro.done();
        }



        struct promise_type {
            promise_type() = default;                               // (1)

            ~promise_type() = default;

            //�������
            std::suspend_always initial_suspend() {                                 // (4)
                return std::suspend_always();
            }
            std::suspend_always  final_suspend() noexcept {
                return std::suspend_always();
            }
            Generator<T>  get_return_object() {                               // (2)
                return Generator<T>{ std::coroutine_handle<promise_type>::from_promise(*this) };
            }

            //co_return�����������co_routine����Ĭ�ϵ���
            void return_value(const T& value) {

            }

            // co_yield���
            std::suspend_always yield_value(const T value) {                        // (6) 
                current_value = value;
                return std::suspend_always{};
            }

            //�̶����
            void unhandled_exception() {
                std::exit(1);
            }

            //�Զ������
            T current_value;
        };

    };

    //ʹ��yield value�������Generator������������ֵ���ã������൱��һ����װ
    Generator<int> getNext(int start = 0, int step = 1) noexcept {
        auto value = start;
        for (int i = 0;; ++i) {
            co_yield value;//��佫��ǰ�� value ���ݸ������ߣ�promise�ڣ�������ʱ����Э�̡���Э�̱��ָ�ʱ��ִ�д� co_yield ֮�����������
            value += step;
        }
    }

    void run() {

        std::cout << std::endl;

        std::cout << "getNext():";
        auto gen = getNext();
        for (int i = 0; i <= 10; ++i) {
            gen.next();//�ָ��߳�
            //��getNext()�У���������Զ������̣߳���ֵ����promise��current_value
            std::cout << " " << gen.getValue();    //��ȡcurrent_value
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