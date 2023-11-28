#pragma once
# include<iostream>
# include<future>

namespace future_asynchronous {
	bool wait_the_data_downloading() {
		//������һ��������̣��������ǲ�����sleep_forģ�⣬��Ϊ����û��ָ��async��launchģʽ���ͼ�װ��Ҫ5��
		return true;
	}

	void do_other_calculations() {
		std::cout << "doing other stuff" << std::endl;
	}

	void run() {
		//Aquire the future associated with that particular asynchronous
		std::future<bool> the_answer_future = std::async(wait_the_data_downloading);
		do_other_calculations();
		std::cout << "The downloading is complete?  " << the_answer_future.get() << std::endl;

	}
}
