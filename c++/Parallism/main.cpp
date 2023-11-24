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

	ship_example_with_queue::captain();
}