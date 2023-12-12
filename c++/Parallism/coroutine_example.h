#pragma once

# include<coroutine>
# include<iostream>
# include<cassert>

class resumable {
	//�Զ���resume�࣬coroutine���̷������rusume���Ͷ���
	//������co_routine handle(wrapped coroutine handle)
	//�˾������
public:
	struct promise_type; //�û�������࣬������Э�̵���Ϊ�������ʼ�������𣬻ָ�������Э��
	using coro_handle = std::coroutine_handle<promise_type>;//�򵥵ľ����ģ�廯Э�̾�������ж�Э�̵Ŀ���Ȩ��promise_typeָ�����������������ĸ����͵�Э�̣���ͬ��promise������Ϊģ������ľ��
	
	resumable(coro_handle handle) :handle_(handle) { assert(handle); };//���캯�����������Ϊ������ȷ���������Ч��
	resumable(resumable&) = delete;//����copy constructor
	resumable(resumable&&) = delete;//����move constructor

	bool resume() {//act as an API for caller code to resume the work using coroutine handle
		if (not handle_.done())//���Э����δ���
			handle_.resume();//�ָ�Э�̵�ִ��
		return not handle_.done();//����Э���Ƿ���Ȼ��ִ��
	}
	~resumable() {
		handle_.destroy();//����Э�̾��
	}
private:
	coro_handle handle_;//Э�̾��ʵ��

};

struct resumable::promise_type {
	using coro_handle = std::coroutine_handle<promise_type>;
	auto get_return_object() {
		return coro_handle::from_promise(*this);
	}
	auto initial_suspend() { return std::suspend_always(); }
	auto final_suspend() noexcept { return std::suspend_always(); }
	void return_void() {}//��Ϊfoo()����return void
	void unhandled_exception() {
		std::terminate();
	}
	
};


resumable foo() {
	std::cout << "a" << std::endl;
	co_await std::suspend_always();	//suspension point �����е��˴���ͣ
	std::cout << "b" << std::endl;
	co_await std::suspend_always();
	std::cout << "c" << std::endl;

	//coroutine������û�а����κη�����䣬������Ϊ���ض����ɱ�������ʽ������������һ��co_coutine handle object
}

void run_coroutine_example() {
	resumable res1 = foo();
	//��ʱ��Э��res1����ʱ���ڹ���״̬
	//�ھ���ϵ��ûָ������Իָ���ִ�С�
	std::cout << "��һ�Σ�Э�̱�����" << std::endl;
	res1.resume();
	std::cout << "���Ѻ󣬵ڶ��α�����" << std::endl;
	res1.resume();
	std::cout << "���Ѻ󣬵����α�����" << std::endl;
	res1.resume();
}