#pragma once
# include <stack>
# include <iostream>
# include <thread>
# include <mutex>
# include <memory>

namespace thread_safe_stack {
	template<typename T>
	class trivial_thread_safe_stack {
		std::stack<T> stk;
		std::mutex m;
	public:
		void push(T element) {
			std::lock_guard<std::mutex> lg(m);
			stk.push(element);
		}
		void pop() {
			std::lock_guard<std::mutex> lg(m);
			stk.pop(element);
		}

		T &top() {
			std::lock_guard<std::mutex> lg(m);
			return stk.top();
		}

		bool empty() {
			std::lock_guard<std::mutex> lg(m);
			return stk.empty();
		}
		size_t size() {
			std::lock_guard<std::mutex> lg(m);
			return stk.size();
		}



	};



	// û�нӿڼ̳о��������İ汾
	template<typename T>
	class trivial_thread_safe_stack {
		std::stack<std::shared_ptr<T>> stk;
		std::mutex m;
	public:
		void push(T element) {
			std::lock_guard<std::mutex> lg(m);
			stk.push(std::make_shared<T>(element));
		}
		std::shared_ptr<T> pop() {
			std::lock_guard<std::mutex> lg(m);
			if (stk.empty()) {
				throw std::runtime_error("stack is empty");
			}

			std::shared_ptr<T> res(stk.top()); // ����ʹ��Ȩ
			stk.pop();// stk.pop()��ָ��û�ˣ�usercount�½�������res��Ȼ��
			return res;
		}

		void pop(T& value) {
			std::lock_guard<std::mutex> lg(m);
			if (stk.empty()) throw std::runtime_error("stack is empty");
			value = *(stk.top().get());
			stk.pop();

		}


		bool empty() {
			std::lock_guard<std::mutex> lg(m);
			return stk.empty();
		}
		size_t size() {
			std::lock_guard<std::mutex> lg(m);
			return stk.size();
		}



	};
}