#pragma once

# include<coroutine>
# include<iostream>
# include<cassert>

class resumable {
	//自定义resume类，coroutine例程返回这个rusume类型对象
	//包含了co_routine handle(wrapped coroutine handle)
	//此句柄将从
public:
	struct promise_type; //用户定义的类，定义了协程的行为，例如初始化，挂起，恢复，销毁协程
	using coro_handle = std::coroutine_handle<promise_type>;//简单的句柄，模板化协程句柄，持有堆协程的控制权，promise_type指定了这个句柄将操作哪个类型的协程，等同于promise类型作为模板参数的句柄
	
	resumable(coro_handle handle) :handle_(handle) { assert(handle); };//构造函数，将句柄作为参数。确保句柄是有效的
	resumable(resumable&) = delete;//禁用copy constructor
	resumable(resumable&&) = delete;//禁用move constructor

	bool resume() {//act as an API for caller code to resume the work using coroutine handle
		if (not handle_.done())//如果协程尚未完成
			handle_.resume();//恢复协程的执行
		return not handle_.done();//返回协程是否仍然在执行
	}
	~resumable() {
		handle_.destroy();//销毁协程句柄
	}
private:
	coro_handle handle_;//协程句柄实例

};

struct resumable::promise_type {
	using coro_handle = std::coroutine_handle<promise_type>;
	auto get_return_object() {
		return coro_handle::from_promise(*this);
	}
	auto initial_suspend() { return std::suspend_always(); }
	auto final_suspend() noexcept { return std::suspend_always(); }
	void return_void() {}//因为foo()函数return void
	void unhandled_exception() {
		std::terminate();
	}
	
};


resumable foo() {
	std::cout << "a" << std::endl;
	co_await std::suspend_always();	//suspension point ：运行到此处暂停
	std::cout << "b" << std::endl;
	co_await std::suspend_always();
	std::cout << "c" << std::endl;

	//coroutine百年身没有包含任何返回语句，这是因为返回对象将由编译器隐式创建，并且是一个co_coutine handle object
}

void run_coroutine_example() {
	resumable res1 = foo();
	//此时，协程res1创建时处于挂起状态
	//在句柄上调用恢复函数以恢复其执行。
	std::cout << "第一次，协程被挂起" << std::endl;
	res1.resume();
	std::cout << "唤醒后，第二次被挂起" << std::endl;
	res1.resume();
	std::cout << "唤醒后，第三次被挂起" << std::endl;
	res1.resume();
}