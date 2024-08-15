作为刷代码随想录前的复习，列出一些常用的STL 算法实现

目前仅处理C++11的算法，不包括ranges，一些新版本较为优秀的功能会额外标注



### `for_each`

```c++
template< class InputIt, class UnaryFunc >
std::for_each(InputIt first, InputIt last, UnaryFunc f);
```



此处就是取代for的作用，对一个container的`[first,last)`范围内的元素都执行`f`

**此处函数需要是支持移动构造的，否则会产生未定义行为，因此通常是使用一个lambda函数**

```c++
#include <iostream>
#include <vector>
#include <algorithm>

class Functor {
public:
    Functor() = default;
    Functor(Functor&&) noexcept = default; // 移动构造函数
    void operator()(int n) const {
        std::cout << n << ' ';
    }
};

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    // 可移动构造的Functor
    std::for_each(v.begin(), v.end(), Functor());
    
    // lambda函数
    std::for_each(v.begin(), v.end(), [](int n){
        std::cout << n << ' ';
    });
                  
    //或者
    auto print = [](const int& n) { std::cout << n << ' '; };    
    std::for_each(v.cbegin(), v.cend(), print);
    return 0;
}

```







该函数的复杂度相比for并没有改善，就是执行`std::distance(first,last)`次`function`







#### 并行版本

> 在C++17中，for_each允许通过执行策略来指定执行方式
>
> ```c++
> template< class ExecutionPolicy, class ForwardIt, class UnaryFunction >
> void for_each( ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, UnaryFunction f );
> 
> ```
>
> 这里的`ExecutionPolicy`包括
>
> * `std::execution::seq`:算法按顺序在单个线程上执行，保证元素按顺序处理。
> * `std::execution::par`：算法在多个线程上并行执行，不保证顺序，可能会在多个核心上分布工作。
> * `std::execution::par_unseq`：算法在多个线程上并行执行，并且可能利用向量化指令，既不保证顺序也不保证单个线程。
>
> 一个例子
>
> ```c++
> #include <iostream>
> #include <vector>
> #include <algorithm>
> #include <execution>
> 
> int main() {
>     std::vector<int> v = {1, 2, 3, 4, 5};
>     
>     // 顺序执行
>     std::for_each(std::execution::seq, v.begin(), v.end(), [](int& n) { n *= 2; });
>     
>     // 并行执行
>     std::for_each(std::execution::par, v.begin(), v.end(), [](int& n) { n *= 2; });
>     
>     // 并行且非顺序化执行
>     std::for_each(std::execution::par_unseq, v.begin(), v.end(), [](int& n) { n *= 2; });
>     
>     for (const auto& n : v) {
>         std::cout << n << ' ';
>     }
>     
>     return 0;
> }
> 
> ```











### `all_of`,`any_of`,`none_of`

`all_of`,`any_of`,`none_of`分别表示对于范围`[first,last)`

* 范围内所有元素是否都满足特定条件
* 范围内是否有任何元素满足特定条件
* 范围内是否都不满足特定条件



```c++
bool all_of(first, last, UnaryPred p);
bool any_of(first, last, UnaryPred p);
bool none_of(first, last, UnaryPred p);

```





```c++
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    bool all_even = std::all_of(v.begin(), v.end(), [](int i){ return i % 2 == 0; });
    bool any_even = std::any_of(v.begin(), v.end(), [](int i){ return i % 2 == 0; });
    bool none_even = std::none_of(v.begin(), v.end(), [](int i){ return i % 2 == 0; });

    std::cout << "All even: " << all_even << "\n";
    std::cout << "Any even: " << any_even << "\n";
    std::cout << "None even: " << none_even << "\n";

    return 0;
}

```



#### 并行版本

> 同样的也能使用C++17的并行方式，参考for_each，不多赘述
>
> ```c++
> template< class ExecutionPolicy, class ForwardIt, class UnaryPred >
> bool all_of( ExecutionPolicy&& policy,
>              ForwardIt first, ForwardIt last, UnaryPred p );
> 
> 
> template< class ExecutionPolicy, class ForwardIt, class UnaryPred >
> bool any_of( ExecutionPolicy&& policy,
>              ForwardIt first, ForwardIt last, UnaryPred p );
> 
> 
> template< class ExecutionPolicy, class ForwardIt, class UnaryPred >
> bool none_of( ExecutionPolicy&& policy,
>               ForwardIt first, ForwardIt last, UnaryPred p );
> ```
>
> 









### `find`,`find_if`,`find_if_not`:线性搜索

这三个函数都返回一个iterator，指向`[first,last)`中**第一个**满足条件的位置，**如果没找到就指向last**

这几个find都是线性搜索，复杂度为`std::distance(first,last)`相当于取代以下情况

```c++
int arr[] = {1, 2, 3, 4, 5};
int count = 0;
for (int* p = arr; p != arr + 5; ++p) {
    if (*p > 2 && *p % 2 == 0) {
        count++;
    }
}

```



**find**

```c++
template< class InputIt, class T >
InputIt find( InputIt first, InputIt last, const T& value );
```

找到第一个等于`value`的元素

==自定义结构体需要重载`operator==`==

```c++
 std::vector<int> v = {1, 2, 3, 4, 5};
 auto it = std::find(v.begin(), v.end(), 3);
```





**find_if**

```c++
template< class InputIt, class UnaryPred >
InputIt find_if( InputIt first, InputIt last, UnaryPred p );
```

`find_if`找到了第一个UnaryPred返回`true`的iterator

```c++
std::vector<int> v = {1, 2, 3, 4, 5};
    auto it = std::find_if(v.begin(), v.end(), [](int n){ return n % 2 == 0; });
```











**find_if_not**

```c++
template< class InputIt, class UnaryPred >
InputIt find_if_not( InputIt first, InputIt last, UnaryPred q );
```

`find_if_not`返回第一个UnaryPred返回false的iterator

虽然可以通过逻辑反转谓词来使用 `std::find_if` 实现 `find_if_not` 的功能，但直接使用 `std::find_if_not` 更直观和易读，提高代码可维护性。

```c++
std::vector<int> v = {2, 4, 6, 8, 9};
auto it = std::find_if_not(v.begin(), v.end(), [](int n){ return n % 2 == 0; });
```



#### 并行版本

> 三者同样存在并行版本，同for_each













### `find_end`

`std::find_end` 用于查找容器中子序列的最后一次出现。常用于处理字符串匹配、数据序列查找等需要确定某个子序列**最后一次出现位置**的场景。

`std::find_end` 使用**朴素的子序列查找算法，逐个元素比较**。*当匹配子序列的第一个元素时，继续比较后续元素，直到匹配成功或失败*。如果成功，则记录位置并继续查找以确保找到最后一次出现的位置。算法的时间复杂度为 O(N*S)，其中 N 是容器长度，S 是子序列长度。

> ==因为该算法是从起始位置开始遍历，每次发现匹配则更新匹配成功位置，因此效率很低==
>
> 甚至不如一种情况，即从末尾开始匹配，找到即返回





find_end返回一个iterator，这个迭代器指向序列`[sub_first,sub_last)`在序列`[first,last)`中最后出现的位置。

如果没有找到，返回`it==last`

当`[sub_first,sub_last)`为空时，认为没有找到





`find_end`包括两个重载

```c++
template< class ForwardIt1, class ForwardIt2 >
ForwardIt1 find_end( ForwardIt1 first, ForwardIt1 last,
                     ForwardIt2 sub_first, ForwardIt2 sub_last );
```

在Forward1中寻找Forward2作为子序列最后一次出现的位置

最多使用`operator==`比较$S\cdot(N-S+1)$次，其中N为`std::distance(first,last)`，S为`std::distance(sub_first,sub_last)`







```c++
template< class ForwardIt1, class ForwardIt2, class BinaryPred >
ForwardIt1 find_end( ForwardIt1 first, ForwardIt1 last,
                     ForwardIt2 sub_first, ForwardIt2 sub_last,
                     BinaryPred p );
```

这是BinaryPred重载版本，它多了一个对比函数，这个函数返回true or false，用于比较两个序列的元素

具体来说，p返回true则认为元素相同。

```c++
bool pred(const Type1 &a, const Type2 &b);
```

最多使用`p`比较$S\cdot(N-S+1)$次，其中N为`std::distance(first,last)`，S为`std::distance(sub_first,sub_last)`





#### 并行版本

同`for_each`

























