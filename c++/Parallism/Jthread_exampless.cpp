# include "Jthread_exampless.h"
#include <thread>

#include <chrono>
#include <iostream>



using namespace::std::literals;

thread_local interrupt_flag this_thread_flag;
bool interrupt_point()
{
    if (this_thread_flag.is_set())
        return true;
    return false;
}
void do_something()
{
    int counter{ 0 };
    while (counter < 10)
    {
        if (interrupt_point())
        {
            return;
        }
        std::this_thread::sleep_for(0.2s);
        std::cerr << "This is interruptible thread : " << counter << std::endl;
        ++counter;
    }
}

void do_something_else()
{
    int counter{ 0 };
    while (counter < 10)
    {
        std::this_thread::sleep_for(0.2s);
        std::cerr << "This is non-interruptible thread : " << counter << std::endl;
        ++counter;
    }
}

void run()
{

    std::cout << std::endl;
    jthread_local nonInterruptable(do_something_else);
    jthread_local interruptible(do_something);

    std::this_thread::sleep_for(1.0s);
    interruptible.interrupt();
    nonInterruptable.interrupt();

    //std::cout << std::endl;

}