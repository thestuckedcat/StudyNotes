# include<iostream>
# include<queue>
# include<vector>



void bfs(const std::vector<std::vector<int>> & adjacency_list, int begin_node){
    std::queue<int> q{begin_node};
    std::vector<int> visited(adjacency_list.size(),0);
    visited[current_node] = 1;

    std::cout << "The bfs sequence is" << std::endl;
    std::cout << begin_node << " ";
    while(!q.empty())
    {
        int current_node = q.front();
        q.pop();
        std::for_each(adjacency_list[current_node].begin(), adjacency_list[current_node].end(),[&](int a){
        if(!visited[a]){
            std::cout << a << " ";
            q.push(a);
            visited[a] = 1;
        }});

    }
    std::cout << std::endl;
}