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


	template<typename T>
	class lock_free_stack_with_harzard_pointer {

		struct node {
			std::shared_ptr<T> data;//使用shared_ptr
			node* next;

			node(T const& _data) :data{ std::make_shared<T>(_data) } {}
		};

		std::atomic<node*> head;
		

		const int max_hazard_pointers = 100;

		struct hazard_pointer {
			std::atomic<std::thread::id> id;
			std::atomic<void*> pointer;
		};

		hazard_pointer hazard_pointers[max_hazard_pointers];

		class hazard_pointer_manager {
			/*
				每个线程都有一个分配的 hazard pointer manager object，当hazard pointer manager构建时，它将在危hazard pointer list中分配一个条目
			*/
			hazard_pointer* hp;
		
		public:
			hazard_pointer_manager() :hp(nullptr) {
				//find out empty slot in Hazard
				for (unsigned i = 0; i < max_hazard_pointers; ++i) {
					std::thread::id default_id;
					if (hazard_pointers[i].id.compare_exchange_strong(default_id, std::this_thread::get_id()))//防止访问hazard_pointers的race condition
					{
						hp = &hazard_pointers[i];
						break;
					}
				}

				if (!hp) {
					throw std::runtime_error("no hazard pointers available");
				}
			}

			std::atomic<void*>& get_pointer() {
				return hp->pointer;
			}

			~hazard_pointer_manager() {
				//将存入的位置保存给hp，这样下一次就不需要遍历寻找
				hp->pointer.store(nullptr);
				hp->id.store(std::thread::id());
			}

		};

		std::atomic<void*>& get_hazard_pointer_for_current_thread() {
			static thread_local hazard_pointer_manager hz_manager;//allow us to have one manager object for each thread，如果只有static，那么所有的线程会共用，如果只有threadlocal或者什么都没有，那么每次调用就会生成一个新的manager
			return hz_manager.get_pointer();
		}

		bool any_outstanding_hazards(node* p) {
			//检查给定节点对象是否有任何未解决的危险指针
			for (unsigned i = 0; i < max_hazard_pointers; ++i) {
				if (hazard_pointers[i].pointer.load() == p) {
					return true;
				}
				return false;
			}
		}
		
		std::atomic<node*> nodes_to_reclaim;

		void reclaim_later(node* _node) {
			_node->next = nodes_to_reclaim.load();
			while (!nodes_to_reclaim.compare_exchange_weak(_node->next, _node));
		}

		void delete_nodes_with_no_hazards() {
			node* current = nodes_to_reclaim.exchange(nullptr);

			while (current) {
				node* const next = current->next;
				if (!any_outstanding_hazards(current)) {
					delete current;
				}
				else {
					reclaim_later(current);
				}
				current = next;
			}
		}

	public:
		void push(T const& _data) {
			node* const new_node = new node(_data);

			new_node->next = head.load();

			while (!head.compare_exchange_weak(new_node->next, new_node));
		}

		void pop(T& result) {
			
			std::atomic<void*>& hp = get_hazard_pointer_for_current_thread();
			node* old_head = head.load();

			//防止运行到此处时（设置该节点为hazard pointer前），其他线程将这个old_head指向的node删除了，我们将下面改为do_while
			do {
				//set hazard pointer
				hp.store(old_head);
			} while (old_head && !head.compare_exchange_strong(old_head, old_head->next));

			//clear hazard pointer
			hp.store(nullptr);

			std::shared_ptr<T> res;
			if (old_head) {
				res.swap(old_head->data);

				if (any_outstanding_hazards(old_head)) {
					reclaim_later(old_head);
				}
				else {
					delete old_head;
				}

				delete_nodes_with_no_hazards();
			}
		}
	};





	template<typename T>
	class lock_free_stack_ref_counting {
	private:
		struct node;

		struct node_wrapper {
			int external_count;
			node* ptr;
		};

		struct node {
			std::shared_ptr<T> data;
			std::atomic<int> internal_count;
			node_wrapper next;

			node(T const& _data) :data(std::make_shared<T>(_data)), internal_count{ 0 } {}

		};

		std::atomic<node_wrapper> head;






		void increment_head_ref_count(node_wrapper& old_counter) //必须要引用传入，因为old_counter可能会被其他增加reference
		{
			node_wrapper new_counter;//代表结果，old_counter其实是代表输入
			
			//这种操作以后都理解为，head实际上都是在高速变化的，我们通过截取其当前时段的静态状态，来进行操作，然后查看操作这段时间之后，是否这个状态改变了。如果改变了，重新做一遍，如果没变就提交上去。

			do {
				new_counter = old_counter;
				++new_counter.external_count;
			} while (!head.compare_exchange_strong(old_counter, new_counter));//在这里，如果其他wrapper先一步增加reference，那么  old_counter就会被head更新，然后再次尝试+1

			old_counter.external_count = new_counter.external_count;//更新old_counter
		}


	public:
		~lock_free_stack_ref_counting() {

		}

		void push(T const& data) {
			//创建即将push的新node，此处是node wrapper
			node_wrapper new_node;

			//然后为这个node wrapper的node创建新的数据
			new_node.ptr = new node(data);
			
			//表示基础链表中的引用
			new_node.external_count = 1;

			//更新head
			new_node.ptr->next = head.load();
			while (!head.compare_exchange_weak(new_node.ptr->next, new_node));
		}

		std::shared_ptr<T> pop() {
			node_wrapper old_head = head.load();

			while (true)
			{
				increment_head_ref_count(old_head);//此处也更新过old_head，有可能是head wrapper，在这个wrapper中指向一个null的node

				node* const ptr = old_head.ptr;
				if (!ptr) {//如果没东西pop
					return std::shared_ptr<T>();
				}

				if (head.compare_exchange_strong(old_head, ptr->next)) // 如果当前head没有变化，则更新head
				{
					std::shared_ptr<T> res;
					res.swap(ptr->data);

					int const current_external_count = old_head.external_count - 2;//-2减去的是基础列表的引用与当前线程引用

					if (ptr->internal_count.fetch_add(current_external_count) == 0) //内部计数应当与为外部计数的负数
					{
						delete ptr;
					}
					return res;
				}
				else if (ptr->internal_count.fectch_sub(1) == 1) //当前head对象被其他线程更改了，上面的compare_exchange函数会将old_head更新到新的head,这意味着当前线程不再引用该节点，因此我们必须减少internal count
				{
					//进入该block代表internal_count在减之前为0，也就是这条线程是保存引用的最后一个线程
					//这里要特别设置==1
					delete ptr;
				}
			}

		}
	};


}