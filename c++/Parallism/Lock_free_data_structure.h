#pragma once
# include <iostream>
# include <thread>
# include <atomic>

namespace Lock_free_stack {
	/*
	stack last in first out, use a pointer called head to represent the node last in
	the node->next point to the node goes before it
	*/
	template<typename T>
	class regular_stack {
	private:
		struct node {
			T data;
			node* next;

			node(T const& _data) :data{ _data } {}
		};

		node* head;

	public:
		void push(T const& _data) {//race condition，push两次只插入一次
			node* const new_node = new node(_data);
			new_node->next = head;
			head = new_node;//关键语句
		}

		void pop(T& result) {
			node* old_head = head;
			head = old_head->next;
			result = old_head->data;
			delete old_head;
		}
	};

	template<typename T>
	class lock_free_stack {
		struct node {
			std::shared_ptr<T> data;//使用shared_ptr
			node* next;

			node(T const& _data) :data{ std::make_shared<T>(_data) } {}
		};

		std::atomic<node*> head;
	public:
		void push(T const& _data) {
			// 申请新的节点
			node* const new_node = new node(_data);

			//将新的节点与当前stack head建立联系
			new_node->next = head.load();
			
			//在这一步查看head是否被修改
			// 如果其他的thread已经先行插入了，此时这个class的head就会指向新的，只需要比较我们在上一步建立的联系是否正确，就可以正确的更新head了
			// 如果head的值与->next相同，代表没有thread修改，因此head就更新为此节点。
			// 如果不同，那么new_node->next就会被存为当前head的值，这样也更新了当前节点上一步的错误
			// 循环直到赋值成功，这种方法不保证插入顺序，只保证没有race condition
			while(!head.compare_exchange_weak(new_node->next, new_node));
		}

		std::shared_ptr<T> pop() {
			/*
				我们需要保证head指针在我们分配新head和result时没有改变，不然就会出现pop两次，实际只pop一次
				实际上，无论是push还是pop，我们都只需要关注将head赋值给新旧节点，以及改变head这两句，其余的都是可以同时操作的
				也就是说这里两个race condition的冲突关键点只有head的读取与修改
				在大多数场合，我们对一个冲突的variable使用一个新的temporary variable来存储，就可以将race condition限制到变量的读取与写入这两个操作中。
			*/
			node* old_head = head.load();
			while (old_head && !head.compare_exchange_weak(old_head, old_head->next));
			//这里，需要先判断old head，是因为存在一种情况，stack已经只剩一个了，两个线程同时pop，此时晚pop的线程会因为原子操作，old_head会被赋值为当前的head，因为没有节点因此为nullptr，此时old_head->next就是在dereference一个nullptr，这明显是错误的
			//因此，应该首先排除old_head为空的状态


			//在普通实现中，返回给调用者之前，节点已经被从栈中移除，或者只有当前线程持有该节点的引用。如果在返回结果时发生异常，就无法回滚已经完成的更改。（即为在上一级的exception中已经访问不到这个数据了，无法做出操作）
			// 因此，在这里使用shared_ptr，可以在异常发生时确保资源被释放，同时，可以在其他地方留有备份，令在这个函数中出现差错时，该数据还有迹可循。
			return old_head ? old_head->data : std::shared_ptr<T>();
		}
	};





	template<typename T>
	class lock_free_stack_without_memory_leak {
		struct node {
			std::shared_ptr<T> data;//使用shared_ptr
			node* next;

			node(T const& _data) :data{ std::make_shared<T>(_data) } {}
		};

		std::atomic<node*> head;
		std::atomic<int> thread_in_pop;
		std::atomic<node*> to_be_deleted;


		void try_reclaim(node* old_head) {
			if (thread_in_pop == 1) {
				//delete node pointed by old head
				delete old_head;

				node* claimed_list = to_be_deleted.exchange(nullptr);
				if (!--thread_in_pop) {
					delete_nodes(claimed_list);
				}
				else if(calimed_list) {
					node* last = claimed_list;
					while (node* const next = last->next) {
						last = next;
					}
					last->next = to_be_deleted;
					while (!to_be_deleted.compare_exchange_weak(last->next, claimed_list));
				}
			}
			else {
				//add node pointed by old_head to the to_be_deleted list
				old_head->next = to_be_deleted;
				while (!to_be_deleted.compare_exchange_weak(old_head->next));
				--thread_in_pop;
			}

		}

		void delete_nodes(node* nodes) {
			while (nodes) {
				node* next = nodes->next;
				delete nodes;
				nodes = next;

			}
		}

	public:
		void push(T const& _data) {
			node* const new_node = new node(_data);

			new_node->next = head.load();

			while (!head.compare_exchange_weak(new_node->next, new_node));
		}

		std::shared_ptr<T> pop() {
			++thread_in_pop;

			node* old_head = head.load();
			while (old_head && !head.compare_exchange_weak(old_head, old_head->next));
			

			std::shared_ptr<T> res;
			if (old_head) {
				res.swap(old_head->data);
			}

			try_reclaim(old_head);

			return res;
		}
	};





}