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





}