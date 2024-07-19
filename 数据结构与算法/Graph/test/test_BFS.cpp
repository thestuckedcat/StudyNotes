#include"../traversal"
#include<vector>
int main(){
    std::vector<std::vector<int>> adj_list{{1,3},
                                            {0},
                                            {3,8},
                                            {0,4,5,2},
                                            {3,6},
                                            {3},
                                            {4,7},
                                            {6},
                                            {2}};


    bfs(adj_list,0);
}