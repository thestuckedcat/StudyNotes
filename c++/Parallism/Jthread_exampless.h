#pragma once

# include<thread>
# include<atomic>
# include<future>



class interrupt_flag
{
	bool local_flag;
public:
	void set()
	{
		local_flag = true;
	}
	bool is_set()
	{
		return local_flag;
	}
};

extern thread_local interrupt_flag this_thread_flag;

class jthread_local {

	std::thread _thread;
	interrupt_flag* flag;

public:
	template<typename FunctionType>
	jthread_local(FunctionType f)
	{
		std::promise<interrupt_flag*> p;
		_thread = std::thread([f, &p] {
			p.set_value(&this_thread_flag);
			f();
			});
		flag = p.get_future().get();
	}


	void interrupt()
	{
		flag->set();
	}

	~jthread_local()
	{
		if (_thread.joinable())
			_thread.join();
	}
};

bool interrupt_point();


void do_something();
void do_something_else();
void run();