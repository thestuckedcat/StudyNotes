# include <cstdio>
# include <iostream>

using namespace std;

// compare arrays
void compare_arrays(int* a, int* b, int size) {
	for (int i = 0; i < size; i++)
	{
		if (a[i] != b[i])
		{
			printf("Arrays are differnt \n");
			return;
		}
	}
	printf("Arrays are same \n");
}