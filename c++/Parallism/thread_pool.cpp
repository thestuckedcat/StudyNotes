# include "thread_pool.h"

namespace my_thread_pool_with_local_queue {
	thread_local std::unique_ptr<std::queue<function_wrapper>> thread_pool::local_work_queue;
}