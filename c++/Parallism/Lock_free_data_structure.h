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
		void push(T const& _data) {//race condition��push����ֻ����һ��
			node* const new_node = new node(_data);
			new_node->next = head;
			head = new_node;//�ؼ����
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
			std::shared_ptr<T> data;//ʹ��shared_ptr
			node* next;

			node(T const& _data) :data{ std::make_shared<T>(_data) } {}
		};

		std::atomic<node*> head;
	public:
		void push(T const& _data) {
			// �����µĽڵ�
			node* const new_node = new node(_data);

			//���µĽڵ��뵱ǰstack head������ϵ
			new_node->next = head.load();
			
			//����һ���鿴head�Ƿ��޸�
			// ���������thread�Ѿ����в����ˣ���ʱ���class��head�ͻ�ָ���µģ�ֻ��Ҫ�Ƚ���������һ����������ϵ�Ƿ���ȷ���Ϳ�����ȷ�ĸ���head��
			// ���head��ֵ��->next��ͬ������û��thread�޸ģ����head�͸���Ϊ�˽ڵ㡣
			// �����ͬ����ônew_node->next�ͻᱻ��Ϊ��ǰhead��ֵ������Ҳ�����˵�ǰ�ڵ���һ���Ĵ���
			// ѭ��ֱ����ֵ�ɹ������ַ�������֤����˳��ֻ��֤û��race condition
			while(!head.compare_exchange_weak(new_node->next, new_node));
		}

		std::shared_ptr<T> pop() {
			/*
				������Ҫ��֤headָ�������Ƿ�����head��resultʱû�иı䣬��Ȼ�ͻ����pop���Σ�ʵ��ֻpopһ��
				ʵ���ϣ�������push����pop�����Ƕ�ֻ��Ҫ��ע��head��ֵ���¾ɽڵ㣬�Լ��ı�head�����䣬����Ķ��ǿ���ͬʱ������
				Ҳ����˵��������race condition�ĳ�ͻ�ؼ���ֻ��head�Ķ�ȡ���޸�
				�ڴ�������ϣ����Ƕ�һ����ͻ��variableʹ��һ���µ�temporary variable���洢���Ϳ��Խ�race condition���Ƶ������Ķ�ȡ��д�������������С�
			*/
			node* old_head = head.load();
			while (old_head && !head.compare_exchange_weak(old_head, old_head->next));
			//�����Ҫ���ж�old head������Ϊ����һ�������stack�Ѿ�ֻʣһ���ˣ������߳�ͬʱpop����ʱ��pop���̻߳���Ϊԭ�Ӳ�����old_head�ᱻ��ֵΪ��ǰ��head����Ϊû�нڵ����Ϊnullptr����ʱold_head->next������dereferenceһ��nullptr���������Ǵ����
			//��ˣ�Ӧ�������ų�old_headΪ�յ�״̬


			//����ͨʵ���У����ظ�������֮ǰ���ڵ��Ѿ�����ջ���Ƴ�������ֻ�е�ǰ�̳߳��иýڵ�����á�����ڷ��ؽ��ʱ�����쳣�����޷��ع��Ѿ���ɵĸ��ġ�����Ϊ����һ����exception���Ѿ����ʲ�����������ˣ��޷�����������
			// ��ˣ�������ʹ��shared_ptr���������쳣����ʱȷ����Դ���ͷţ�ͬʱ�������������ط����б��ݣ�������������г��ֲ��ʱ�������ݻ��м���ѭ��
			return old_head ? old_head->data : std::shared_ptr<T>();
		}
	};





	template<typename T>
	class lock_free_stack_without_memory_leak {
		struct node {
			std::shared_ptr<T> data;//ʹ��shared_ptr
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
			std::shared_ptr<T> data;//ʹ��shared_ptr
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
				ÿ���̶߳���һ������� hazard pointer manager object����hazard pointer manager����ʱ��������Σhazard pointer list�з���һ����Ŀ
			*/
			hazard_pointer* hp;
		
		public:
			hazard_pointer_manager() :hp(nullptr) {
				//find out empty slot in Hazard
				for (unsigned i = 0; i < max_hazard_pointers; ++i) {
					std::thread::id default_id;
					if (hazard_pointers[i].id.compare_exchange_strong(default_id, std::this_thread::get_id()))//��ֹ����hazard_pointers��race condition
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
				//�������λ�ñ����hp��������һ�ξͲ���Ҫ����Ѱ��
				hp->pointer.store(nullptr);
				hp->id.store(std::thread::id());
			}

		};

		std::atomic<void*>& get_hazard_pointer_for_current_thread() {
			static thread_local hazard_pointer_manager hz_manager;//allow us to have one manager object for each thread�����ֻ��static����ô���е��̻߳Ṳ�ã����ֻ��threadlocal����ʲô��û�У���ôÿ�ε��þͻ�����һ���µ�manager
			return hz_manager.get_pointer();
		}

		bool any_outstanding_hazards(node* p) {
			//�������ڵ�����Ƿ����κ�δ�����Σ��ָ��
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

			//��ֹ���е��˴�ʱ�����øýڵ�Ϊhazard pointerǰ���������߳̽����old_headָ���nodeɾ���ˣ����ǽ������Ϊdo_while
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






		void increment_head_ref_count(node_wrapper& old_counter) //����Ҫ���ô��룬��Ϊold_counter���ܻᱻ��������reference
		{
			node_wrapper new_counter;//��������old_counter��ʵ�Ǵ�������
			
			//���ֲ����Ժ����Ϊ��headʵ���϶����ڸ��ٱ仯�ģ�����ͨ����ȡ�䵱ǰʱ�εľ�̬״̬�������в�����Ȼ��鿴�������ʱ��֮���Ƿ����״̬�ı��ˡ�����ı��ˣ�������һ�飬���û����ύ��ȥ��

			do {
				new_counter = old_counter;
				++new_counter.external_count;
			} while (!head.compare_exchange_strong(old_counter, new_counter));//������������wrapper��һ������reference����ô  old_counter�ͻᱻhead���£�Ȼ���ٴγ���+1

			old_counter.external_count = new_counter.external_count;//����old_counter
		}


	public:
		~lock_free_stack_ref_counting() {

		}

		void push(T const& data) {
			//��������push����node���˴���node wrapper
			node_wrapper new_node;

			//Ȼ��Ϊ���node wrapper��node�����µ�����
			new_node.ptr = new node(data);
			
			//��ʾ���������е�����
			new_node.external_count = 1;

			//����head
			new_node.ptr->next = head.load();
			while (!head.compare_exchange_weak(new_node.ptr->next, new_node));
		}

		std::shared_ptr<T> pop() {
			node_wrapper old_head = head.load();

			while (true)
			{
				increment_head_ref_count(old_head);//�˴�Ҳ���¹�old_head���п�����head wrapper�������wrapper��ָ��һ��null��node

				node* const ptr = old_head.ptr;
				if (!ptr) {//���û����pop
					return std::shared_ptr<T>();
				}

				if (head.compare_exchange_strong(old_head, ptr->next)) // �����ǰheadû�б仯�������head
				{
					std::shared_ptr<T> res;
					res.swap(ptr->data);

					int const current_external_count = old_head.external_count - 2;//-2��ȥ���ǻ����б�������뵱ǰ�߳�����

					if (ptr->internal_count.fetch_add(current_external_count) == 0) //�ڲ�����Ӧ����Ϊ�ⲿ�����ĸ���
					{
						delete ptr;
					}
					return res;
				}
				else if (ptr->internal_count.fectch_sub(1) == 1) //��ǰhead���������̸߳����ˣ������compare_exchange�����Ὣold_head���µ��µ�head,����ζ�ŵ�ǰ�̲߳������øýڵ㣬������Ǳ������internal count
				{
					//�����block����internal_count�ڼ�֮ǰΪ0��Ҳ���������߳��Ǳ������õ����һ���߳�
					//����Ҫ�ر�����==1
					delete ptr;
				}
			}

		}
	};


}