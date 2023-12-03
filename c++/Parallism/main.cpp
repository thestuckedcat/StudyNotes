# include "launch_a_thread.h"
# include <iostream>
# include <vector>
# include "join_joinable_detach.h"
# include "exception_scenario_join.h"
# include "ship_example.h"
# include "pass_parameters.h"
# include "transferring_thread_ownership.h"
# include "useful_functions_in_thread.h"
# include "ship_example2.h"
# include "parallel_accumulation.h"
# include "thread_local_storage.h"
# include "Using_mutex.h"
# include "dead_lock.h"
# include "unique_locks.h"
# include "Drive_example.h"
# include "thread_safe_queue.h"
# include "future_asynchronous.h"
# include "deep_dive_in_async.h"
# include "accumulate_with_async.h"
# include "package_task_example.h"
# include "promise_future_example.h"
# include "promise_send_exception.h"
# include "shared_future_example.h"
# include "Using_Parallel_STL.h"


int main() {
	//launch_a_thread::run();

	//join_joinable_detach::run_join_joinable();

	//join_joinable_detach::run_join_detach();

	//exception_scenario::run();


	//std::vector<int> input {1, 2, 3, 1, 1, 2, 3, 4, 100, 1};
	//ship_example::captain(input);

	//pass_parameters::run_1();
	//pass_parameters::run_2();

	//transfer_ownership::run();

	//useful_functions::run();

	//ship_example_with_queue::captain();

	//parallel_accumulation::run();

	//thread_local_storage::run();

	//using_mutex::run();

	//dead_lock_example::run_1();
	//dead_lock_example::run_2();

	//unique_lock_example::run_1();
	//unique_lock_example::run_2();


	//drive_example::run_2();

	//thread_safe_queue_space::run();

	//future_asynchronous::run();

	//deep_dive_in_async::run();

	//accumulate_with_async::run();

	//package_task_example::run();
	
	//promise_future_example::run();
	//promise_future_example::run_with_package();

	//promise_send_exception::run();

	//shared_future_example::run_2();

	using_parallel_stl::run();
}