## 1. Introduction

C++标准模板库（STL）提供了一系列的容器，以支持各种数据结构的需求。这些容器可以分为几类：序列容器、关联容器、无序关联容器和容器适配器，每种容器都有其独特的用途和性能特点，适合不同的编程场景。

### 序列容器

序列容器如 `vector`, `deque`, `array`, `list`, 和 `forward_list` 提供了一种按顺序存储元素的方式。它们各有优缺点：

- **`vector`** ：变长数组，高效的尾部插入，O(1)的随机访问，但是在除了尾部的其他地方插入删除元素较慢。
- **`deque`**：双端堆列，它使用链表形式连接很多个小数组，因此支持在头部和尾部快速插入和删除，但是因此不支持O(1)的随机访问。 
- **`array`**：提供了固定大小的存储，与内置数组性能相当，但更安全。
- **`list`** 是一个双向链表，任何位置的插入和删除都很高效，但不支持快速随机访问。
- **`forward_list`** 是C++11引入的单向链表，相比list更加优化空间使用，只能向前遍历。



### 关联容器与无序关联容器

关联容器自动根据特定的排序准则（元素的键值）排序元素，并允许快速查找（通常为O(logn)）。他们通常实现为拥有平衡操作的二叉搜索树（红黑树）。

- **`set`**：集合，包含**键**的**有序集**，==键唯一==。
- **`multiset`**：集合，包含**键**的**有序集**，键可以重复。
- **`map`**：映射，包含**键值对**的**有序集**，==键唯一==。
- **`multimap`**：映射，包含**键值对**的**有序集**，键可以重复。

> 这些容器用于需要大量查找操作的场合，能够提供高效的搜索、删除和插入操作。



无序关联容器不会根据元素的键值自动排序，而是使用哈希表来存储元素，提供平均常数时间复杂度的查找、插入和删除操作。

- **`unordered_set`**（C++11）：一个使用哈希表实现的集合，键唯一。
- **`unordered_multiset`**（C++11）：一个使用哈希表实现的集合，键可以重复。
- **`unordered_map`**（C++11）：一个使用哈希表实现的映射，键唯一。
- **`unordered_multimap`**（C++11）：一个使用哈希表实现的映射，键可以重复。

> 无序关联容器因为使用哈希表，因此能够快速访问元素。

这些容器的引入，扩展了C++在数据处理和性能优化上的能力，特别是在处理复杂数据集和实现高效算法方面。





### 容器适配器

容器适配器提供了一种方式，通过特定的接口限制对底层容器的访问方式，从而实现特定的数据结构（如栈、队列等）。它们并不是自成一体的容器，而是在现有容器接口上提供了限制。常见的容器适配器包括：

- **`stack`** 提供了**后进先出**（LIFO）的数据访问模型。
- **`queue`** 提供了**先进先出**（FIFO）的数据访问模型。
- **`priority_queue`** 根据元素优先级出队，支持动态的优先级队列。

这些适配器通过封装基础容器（如 `vector`, `deque`）并限制其接口来实现特定的行为，简化了复杂数据结构的实现，让开发者能够更专注于业务逻辑。



### `string`

最后，`string` 容器是一个专门用于字符存储和操作的类，实质上像是一个 `vector<char>`。但它提供了大量的成员函数来支持字符串处理，如查找、替换、截取等操作。`string` 的设计极大地方便了文本数据的处理，并且由于其动态数组的底层实现，`string` 也支持高效的随机访问和动态扩展。

































## 2. 序列容器

### 2.0 容器关系简介

#### 序列容器的分类

序列容器根据其内部数据结构可分为两类：**数组基序列容器**（如 `vector`, `deque`, `array`）和 **链表基序列容器**（如 `list`, `forward_list`）。

#### 数组基序列容器

这类容器的主要特点是支持快速随机访问，操作包括：

- **容量相关操作**：如 `resize()`, `reserve()`, 调整容器大小和预留内存。
- **访问头尾元素**：通过 `front()` 和 `back()` 访问首尾元素。
- **随机插入与删除**：
  - 插入（`insert(pos, value)`）总在给定位置 `pos` 前进行，这样就能通过.begin()插入序列头部。
  - 删除（`erase(begin, end)`）涵盖 `[begin, end)` 范围，避免了删除 `end()` 位置的无效操作。
- **向容器尾部或头部插入和删除**：操作如 `push_back()`, `pop_back()`, `push_front()`, `pop_front()` 提供对容器两端的快速操作。

#### 链表基序列容器

链表容器提供了优化的插入和删除操作，但不支持快速随机访问：

- **不支持 `std::sort()`**：链表使用自定义的 `sort()` 方法，适应其非连续内存结构。
- **链表特有操作**：
  - **Merge**：优化的合并操作，合并两个已排序的链表。
  - **Splice**：`insert()` 的移动版本，将元素从一个链表转移至另一个，修改了原链表的结构。
- **针对 `forward_list`**：设计了 `insert_after`, `splice_after`, `erase_after` 等操作，以适应单向链表的特性，提高了在已知节点后的操作效率。

#### 总结

数组基容器适用于需要快速随机访问和频繁操作尾部元素的场景，而链表基容器则适合元素插入和删除操作更为频繁的情况，尤其是在元素位置不固定时。每种容器类型的设计都旨在最大化特定操作的效率，适应不同的应用需求。

























### 2.1 `vector`

#### Vector的构造原理

`vector` 的主要特点是能够根据需要动态地调整容器的大小。为了实现这一点，它在内部使用一个动态数组来存储其元素。当新元素插入并超出当前容量时，`vector` 会：

1. 分配一个更大的数组（通常是当前容量的两倍，这一点依赖于具体实现）。
2. 将旧数组中的元素复制到新数组中。
3. 销毁旧数组并更新内部结构以使用新数组。

这个“增长因子”通常是2，意味着每次容量增加都会加倍，这是一种折中的选择，旨在平衡内存使用和复制开销。



#### **构造vector**

```c++
#include <vector>

std::vector<int> v1;                                // 空的vector
std::vector<int> v2(5, 10);                         // 包含5个元素，每个元素的值都是10
std::vector<int> v3(v2.begin(), v2.end());          // 使用迭代器范围构造
std::vector<int> v4(v3);                            // 拷贝构造函数
std::vector<int> v5(std::move(v4));                 // 移动构造函数
std::vector<int> v6 {1, 2, 3, 4, 5};                // 初始化列表构造函数
```





#### 查

- **`front()`** - 返回对第一个元素的引用。时间复杂度为 O(1)。
- **`back()`** - 返回对最后一个元素的引用。时间复杂度为 O(1)。
- **`data()`** - 返回指向内存中第一个元素的指针。时间复杂度为 O(1)。



- **`empty()`** - 检查容器是否为空。时间复杂度为 O(1)。
- **`size()`** - 返回容器中的元素数。时间复杂度为 O(1)。
- **`max_size()`** - 返回容器可能容纳的最大元素数。时间复杂度为 O(1)。==容器理论最大元素数量==，由系统地址空间决定。
- **`reserve(size_type n)`** - 请求改变容器容量至少足以容纳 `n` 个元素。时间复杂度为 O(n)，因为可能触发重分配。
- **`capacity()`** - 返回当前为容器分配的内存中可以存储的元素数量，不必等同于元素实际数量。时间复杂度为 O(1)。





#### 改

- **`clear()`** - 清空容器中的所有元素。时间复杂度为 O(n)。不会改变capacity。

- 任意位置==pos前==插入，删除==单个值==

  - **`insert(const_iterator pos, const T& value)`** - 在指定位置==之前==插入元素。时间复杂度平均为 O(n)。
  - **`emplace(const_iterator pos, Args&&... args)`** - 构造元素就地以减少复制或移动操作的数量。时间复杂度平均为 O(n)。同样是在之前插入。
  - **`erase(const_iterator pos)`** - 删除指定位置的元素。时间复杂度平均为 O(n)。

- 任意位置==pos前==插入，删除多个值

  - **`insert(iterator pos, to_insert.begin(), to_insert.end());`**

  - **`erase(this.begin(),this.begin() + k)`**

    ```c++
    std::vector<int> v = {10, 20, 30};
    std::vector<int> to_insert = {40, 50, 60};
    v.insert(v.begin() + 1, to_insert.begin(), to_insert.end());
    //10 40 50 60 20 30 
    
     std::vector<int> v = {10, 40, 50, 60, 20, 30};
    // 删除多个元素
    v.erase(v.begin() + 1, v.begin() + 4);
    // 10 20 30 
    ```

    

- 尾端操作

  - **`push_back(const T& value)`** - 在容器尾部添加一个元素。时间复杂度为摊销 O(1)。
  - **`emplace_back(Args&&... args)`** - 在容器尾部就地构造一个元素。时间复杂度为摊销 O(1)。
  - **`pop_back()`** - 删除容器尾部的元素。时间复杂度为 O(1)。

- **`resize(size_type n)`** - 调整容器的大小，新元素会被初始化。时间复杂度为 O(n)。

- **`swap(vector& other)`** - 与另一个 `vector` 交换内容。时间复杂度为 O(1)。

- 考虑到`push_front`耗时太大，因此vector不考虑任何头部操作。

> (Args&&... args)是完美转发，它直接接受打包的参数。
>
> ```c++
> std::vector<std::pair<int, std::string>> vec;
> 
> vec.emplace_back(2, "Banana");
> ```



















### 2.2 deque

#### Deque的构造原理

`deque` 的内部通常不像 `vector` 那样仅使用一个连续的内存区域。而是使用一个中心控制器来维护多个固定大小的数组（缓冲区），每个数组可以存储多个元素。`deque` 的这种实现方式支持==在两端快速添加或删除元素==，而无需像 `vector` 那样经常进行内存重新分配。





#### Deque的常用构造方法

```c++
std::deque<int> d1;                                // 空的deque
std::deque<int> d2(5, 10);                         // 包含5个元素，每个元素的值都是10
std::deque<int> d3(d2.begin(), d2.end());          // 使用迭代器范围构造
std::deque<int> d4(d3);                            // 拷贝构造函数
std::deque<int> d5(std::move(d4));                 // 移动构造函数
std::deque<int> d6 {1, 2, 3, 4, 5};                // 初始化列表构造函数
```







#### 查

- **`front()`** - 返回对第一个元素的引用。时间复杂度为 O(1)。
- **`back()`** - 返回对最后一个元素的引用。时间复杂度为 O(1)。
- `deque` 没有 `data()` 方法，因为其内部数据不是连续存储的。



- **`empty()`** - 检查容器是否为空。时间复杂度为 O(1)。
- **`size()`** - 返回容器中的元素数。时间复杂度为 O(1)。
- **`max_size()`** - 返回容器可能容纳的最大元素数。时间复杂度为 O(1)。
- `deque` 没有 `reserve()` 和 `capacity()` 方法，因为其复杂的内部结构不支持像 `vector` 那样的简单容量预留。







#### 改

- **`clear()`** - 清空容器中的所有元素。时间复杂度为 O(n)。

- 删除，pos位置前插入==单个元素==

  - **`insert(const_iterator pos, const T& value)`** - 在指定位置之前插入元素。时间复杂度平均为 O(n)。
  - **`emplace(const_iterator pos, Args&&... args)`** - 构造元素就地以减少复制或移动操作的数量。时间复杂度平均为 O(n)。
  - **`erase(const_iterator pos)`** - 删除指定位置的元素。时间复杂度平均为 O(n)。

- 删除，pos位置前插入==多个元素==

  - **`insert(pos, to_insert.begin(),to_insert.end())`**

  - **`erase(this.begin(), this.begin() + k)`**

    ```c++
    std::deque<int> d = {10, 20, 30};
    
    // 插入多个元素
    std::deque<int> to_insert = {40, 50, 60};
    d.insert(d.begin() + 1, to_insert.begin(), to_insert.end());
    // 10 40 50 60 20 30 
    
    std::deque<int> d = {10, 40, 50, 60, 20, 30};
    // 删除多个元素
    d.erase(d.begin() + 1, d.begin() + 4);
    // 10 20 30 
    ```

    

- 尾端操作

  - **`push_back(const T& value)`** - 在容器尾部添加一个元素。时间复杂度为 O(1) 平摊。
  - **`emplace_back(Args&&... args)`** - 在容器尾部就地构造一个元素。时间复杂度为 O(1) 平摊。
  - **`pop_back()`** - 删除容器尾部的元素。时间复杂度为 O(1)。

- **`resize(size_type n)`** - 调整容器的大小，新元素会被初始化。时间复杂度为 O(n)。

- **`swap(deque& other)`** - 与另一个 `deque` 交换内容。时间复杂度为 O(1)。

- 首端操作（new)

  - **`push_front(const T& value)`** - 在容器前端添加一个元素。时间复杂度为 O(1) 平摊。
  - **`emplace_front(Args&&... args)`** - 在容器前端就地构造一个元素。时间复杂度为 O(1) 平摊。
  - **`pop_front()`** - 删除容器前端的元素。时间复杂度为 O(1)。



`deque` 设计为支持快速在两端插入或删除元素，这对于某些算法和数据结构（如双向队列和==滑动窗口==）非常有用。这些操作的存在允许 `deque` 作为一个高效的队列使用，其中元素既可以从前端被推入也可以被弹出，这样的操作对于 `vector` 来说成本较高，因为它涉及到数据的大量移动。

deque其实就是构造队列很方便的基础数据结构（stack, queue)，通常不会直接使用deque，而是通过Adaptor使用其衍生容器。





































### 2.3 Array

在 C++ 标准模板库（STL）中，`array` 是一种容器，它封装了固定大小的数组。`array` 提供了类似于传统数组的行为，但同时增加了一些容器的特性，如迭代器支持和标准容器接口。

```c++
template<
    class T,
    std::size_t N
> struct array;
```

#### Array的构造原理

`array` 是一个模板类，提供了对==固定大小==的数组的封装。与普通数组不同的是，`array` 的大小在编译时确定，并且作为模板参数传递给 `array` 类。由于它的大小固定，`array` 提供了编译时安全检查等优点，但也意味着它不具备动态数组的灵活性。



#### Array的常用构造方法

```c++
#include <array>

std::array<int, 5> a1 = {1, 2, 3, 4, 5};           // 列表初始化
std::array<int, 5> a2 {{1, 2, 3, 4, 5}};           // 同上，使用双括号
std::array<int, 5> a3;                             // 默认初始化，每个元素都是未定义的
```



#### 查

- **`front()`** - 返回对第一个元素的引用。时间复杂度为 O(1)。
- **`back()`** - 返回对最后一个元素的引用。时间复杂度为 O(1)。
- **`data()`** - 返回指向内存中第一个元素的指针。时间复杂度为 O(1)。



- **`empty()`** - 检查容器是否为空。对于 `array`，这总是返回 false 除非数组大小为零。时间复杂度为 O(1)。
- **`size()`** - 返回容器中的元素数，即数组的大小。时间复杂度为 O(1)。
- **`max_size()`** - 返回容器可能容纳的最大元素数，同 `size()`。时间复杂度为 O(1)。
- `array` 没有 `reserve()` 和 `capacity()` 方法，因为其大小是固定的，不需要也不能动态分配或调整容量。



#### 改

- **`fill(const T& value)`** - 用给定值填充数组。时间复杂度为 O(n)。fill就类似`void* memset(void* s, int c, size_t n);`
- **`swap(array& other)`** - 与另一个 `array` 的内容交换，只要它们的类型和大小相同。时间复杂度为 O(n)。



`array` 没有如 `clear`, `insert`, `emplace`, `erase`, `push_back`, `emplace_back`, `pop_back`, `resize`，以及任何前端操作（`push_front`, `emplace_front`, `pop_front`）的方法，因为：

- **固定大小**：`array` 的大小在编译时确定，==不能增加或减少==。
- **静态结构**：作为一个静态大小的数组封装，其目的是提供对底层原始数组的简单封装和安全访问，而不是提供动态容器的灵活性。



























### 2.4 list

在 C++ 标准模板库（STL）中，`list` 是一种序列容器，它通过内部的==双向链表==实现。这使得在列表中任何位置的插入和删除操作都非常高效。

#### List的构造原理

`list` 使用节点的双向链表来存储数据。每个节点包含三个基本元素：数据字段、指向前一个节点的指针和指向下一个节点的指针。由于数据不是连续存储的，`list` 支持在常数时间内对任何位置的元素进行快速插入和删除。



#### List的常用构造方法

```c++
std::list<int> l1;                               // 空的list
std::list<int> l2(5, 10);                        // 包含5个元素，每个元素的值都是10
std::list<int> l3(l2.begin(), l2.end());         // 使用迭代器范围构造
std::list<int> l4(l3);                           // 拷贝构造函数
std::list<int> l5(std::move(l4));                // 移动构造函数
std::list<int> l6 {1, 2, 3, 4, 5};               // 初始化列表构造函数
```







#### 查

- **`front()`** - 返回对第一个元素的引用。时间复杂度为 O(1)。
- **`back()`** - 返回对最后一个元素的引用。时间复杂度为 O(1)。



- **`empty()`** - 检查容器是否为空。时间复杂度为 O(1)。
- **`size()`** - 返回容器中的元素数。通常是 O(1)，但在某些实现中可能为 O(n)。
- **`max_size()`** - 返回容器可能容纳的最大元素数。时间复杂度为 O(1)。





#### 改

- **`clear()`** - 清空容器中的所有元素。时间复杂度为 O(n)。

- 删除pos位置的值， pos前插入值

  - **`insert(const_iterator pos, const T& value)`** - 在指定位置之前插入元素。时间复杂度为 O(1)，但找到插入点可能是 O(n)。
  - **`emplace(const_iterator pos, Args&&... args)`** - 在指定位置就地构造元素，减少复制或移动操作的数量。时间复杂度为 O(1)。
  - **`erase(const_iterator pos)`** - 删除指定位置的元素。时间复杂度为 O(1)。

- 删除多个值，pos前插入多个值

  - **`insert(const_iterator pos, InputIterator first, InputIterator last);`**
  - **`erase(const_iterator first, const_iterator last);`**

- 尾部操作

  - **`push_back(const T& value)`** - 在容器尾部添加一个元素。时间复杂度为 O(1)。
  - **`emplace_back(Args&&... args)`** - 在容器尾部就地构造一个元素。时间复杂度为 O(1)。
  - **`pop_back()`** - 删除容器尾部的元素。时间复杂度为 O(1)。

- 首部操作

  - **`push_front(const T& value)`** - 在容器前端添加一个元素。时间复杂度为 O(1)。

  - **`emplace_front(Args&&... args)`** - 在容器前端就地构造一个元素。时间复杂度为 O(1)。

  - **`pop_front()`** - 删除容器前端的元素。时间复杂度为 O(1)。

- **`resize(size_type n)`** - 调整容器的大小，新元素会被初始化。如果 n 小于当前大小，则元素将被删除；如果 n 大于当前大小，则将添加默认元素。时间复杂度为 O(n)。
- **`swap(list& other)`** - 与另一个 `list` 交换内容。时间复杂度为 O(1)。











#### 链表操作

* **`merge(list& other, Compare comp)`**:

  **.自定义排序逻辑：** 默认情况下，`sort()` 和 `merge()` 使用元素类型的 `<` 操作符来比较元素。如果元素类型没有自然的小于关系，或者用户需要一个不同的排序准则（例如，按字符串长度而非字典顺序排序），则可以通过 `compare` 函数来指定。

  **统一合并和排序标准：** 在使用 `merge()` 方法合并两个已排序的列表时，很重要的一点是这两个列表应该使用相同的排序标准。这意味着在对列表进行排序和合并时使用的 `compare` 函数应该相同，以确保合并的结果仍然是有序的。

  以下行为会导致未定义的顺序：

  * 两个list有任意一个没有sort
  * sort的compare function与merge的不同

==splice是移动other list，insert是复制other list==

* `splice`：链表重链接

  * **`splice(iterator pos, list& other)`**
    * 将other中所有元素转移到当前列表，插入到pos之前

  * **`splice(iterator pos, list& other, const_iterator it)`**
    * 将other中it上的元素插入pos前

  * **`void splice(const_iterator pos, list& other, const_iterator first, const_iterator last);`**
    * 将 `other` 列表中从 `first` 到 `last`（不包括 `last`）的元素转移到当前列表中，插入位置在 `pos` 指向的元素之前。

* **`remove`**

  * **`remove(const T&value);`**
    * 移除所有等于T的值

  * **`remove_if(Comp compare)`**

    * ```c++
      std::list<int> mylist = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      mylist.remove_if([](int n){ return n % 2 == 0; });  // 移除所有偶数
      ```

* **`reverse()`**
  * 列表翻转



* **`unique(Comp compare)`**

  * 移除列表所有连续重复的元素只留下一个,如果你没有sort，那么只能移除相邻重复元素，这通常没什么用。

  * 如果传入compair，就移除compare指定的规则

  * ```c++
    std::list<int> mylist = {1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4};
    mylist.unique();
    
    // 结果: mylist = {1, 2, 3, 4}
    
    
    mylist.unique([](int first, int second) { return first == second; });
    
    ```





* **`sort(Comp compare)`**

  * `sort()` 方法对列表中的元素进行排序。==注意 `std::list` 使用自己的 `sort()` 方法而非 `std::sort`，因为 `std::list` 不能随机访问其元素。==

  * ```c++
    std::list<int> mylist = {4, 1, 8, 3, 6, 5, 7, 2};
    mylist.sort();  // 默认升序排序
    
    // 结果: mylist = {1, 2, 3, 4, 5, 6, 7, 8}
    
    mylist.sort([](int a, int b) { return a > b; });  // 降序排序
    
    // 结果: mylist = {8, 7, 6, 5, 4, 3, 2, 1}
    
    ```

















### 2.5 forward_list

在 C++ 标准模板库（STL）中，`forward_list` 是一种序列容器，它通过内部的单向链表实现。`forward_list` 是在 C++11 中引入的，以提供比 `std::list` 更高效的空间利用率，因为它仅使用单个链接。以下是关于 `forward_list` 的详细笔记，包括其构造原理、常用构造方法以及常见方法的作用和复杂度。

#### ForwardList的构造原理

`forward_list` 实现为一个单向链表，其中每个元素只包含数据和一个指向下一个元素的指针。这种设计比双向链表（如 `std::list`）使用的内存更少，因为它不需要存储指向前一个元素的指针。这使得 `forward_list` 在空间使用上非常高效，但也意味着它只支持单向顺序访问。

#### ForwardList的常用构造方法

```c++
#include <forward_list>

std::forward_list<int> fl1;                               // 默认构造一个空的forward_list
std::forward_list<int> fl2(5, 10);                        // 包含5个初始化为10的元素
std::forward_list<int> fl3(fl2.begin(), fl2.end());       // 使用迭代器范围构造
std::forward_list<int> fl4(fl3);                          // 拷贝构造函数
std::forward_list<int> fl5(std::move(fl4));               // 移动构造函数
std::forward_list<int> fl6 {1, 2, 3, 4, 5};               // 初始化列表构造函数
```

#### 方法的作用与复杂度

##### Element Access

- **`front()`** - 返回对第一个元素的引用。时间复杂度为 O(1)。
- `forward_list` 没有 `back()` 方法，因为作为一个单向链表，访问最后一个元素需要遍历整个列表，这与 `forward_list` 的设计目标（高效操作）相矛盾。

##### Capacity

- **`empty()`** - 检查容器是否为空。时间复杂度为 O(1)。
- `forward_list` 没有 `size()` 方法，因为计算大小需要遍历整个列表，这会是一个 O(n) 操作，不符合 `forward_list` 设计的高效性原则。
- **`max_size()`** - 返回容器可能容纳的最大元素数。时间复杂度为 O(1)。

##### Modifiers

- **`clear()`** - 清空容器中的所有元素。时间复杂度为 O(n)。
- 特别的==insert_after，erase_after, emplace_after==
  - forward_list只有insert_after，因为由于每个节点没有指向前一个节点的指针，这就意味着你不能直接在某个节点前插入一个新节点，除非你已经持有前一个节点的引用。这与双向链表（如 `std::list`）不同，双向链表的节点既有指向前一个节点的指针也有指向后一个节点的指针，使得直接在任何节点前插入成为可能。
  - 同理，erase不能删除这个节点，因为你无法将父节点与子节连接，你只能删除子节点。
  - **`insert_after(const_iterator pos, const T& value)`** - 在指定位置后插入一个元素。时间复杂度为 O(1)。
  - **`emplace_after(const_iterator pos, Args&&... args)`** - 在指定位置后就地构造一个元素。时间复杂度为 O(1)。
  - **`erase_after(const_iterator pos)`** - 删除指定位置后的一个元素。时间复杂度为 O(1)。
- 但是，考虑到你可以通过iterator访问到`before_begin` iterator，因此支持头部插入。
  - **`push_front(const T& value)`** - 在容器前端添加一个元素。时间复杂度为 O(1)。
  - **`emplace_front(Args&&... args)`** - 在容器前端就地构造一个元素。时间复杂度为 O(1)。
  - **`pop_front()`** - 删除容器前端的元素。时间复杂度为 O(1)。
- **`resize(size_type n)`** - 调整容器的大小，新元素将被默认初始化。时间复杂度为 O(n)。
- **`swap(forward_list& other)`** - 与另一个 `forward_list` 交换内容。时间复杂度为 O(1)。





##### Operations

与list基本相同

- **`merge(forward_list& other)`** - 合并两个已排序的 `forward_list`。时间复杂度为 O(n)。
- **`splice_after(const_iterator pos, forward_list& other)`** - 将另一个 `forward_list` 的元素接在指定位置之后。时间复杂度取决于移动的元素数量，基本操作为 O(1)。
- **`remove(const T& value)`** - 移除所有等于指定值的元素。时间复杂度为 O(n)。
- **`remove_if(UnaryPredicate pred)`** - 移除满足条件的元素。时间复杂度为 O(n)。
- **`reverse()`** - 反转 `forward_list`。时间复杂度为 O(n)。
- **`unique()`** - 移除连续并相等的元素，只保留一个。时间复杂度为 O(n)。
- **`sort()`** - 对 `forward_list` 进行排序。通常时间复杂度为 O(n log n)。





#### 注意事项

==注意，考虑到forward_list不能在pos之前插入，因此对于其在尾部插入有一些特殊的地方==

* 你不能使用.end(),.end()实际只是起到一个结束标识符的作用，`.end()-1`是不合法的

* ```c++
   // 找到 flist1 的最后一个节点的前一个节点
  auto it = flist1.before_begin();
  for (auto next = flist1.begin(); next != flist1.end(); next = std::next(next)) 
  {
      it = next;
  }
  // 将 flist2 的所有元素移动到 flist1 的尾部
  flist1.splice_after(it, flist2);
  ```









## 3. 关联容器

### 3.0 关联容器简介

##### “查” 中的改变： look up 函数

关联容器删除了容器容量的改变与首尾的概念（reserve, capacity, front, back, data）

不像序列容器，关联容器本身就是一个==有序集==，同时也不像序列容器的序列状态，关联容器通常是一个树的状态，这也导致了关联容器不像序列容器那样有明显的首尾概念(front, back)。

例如，在 set 中，可以使用 `*begin()` 和 `*rbegin()` 来访问排序最小和最大的元素，但这不同于 `front()` 和 `back()`，后者暗示了序列的物理布局。



关联容器（如 `set`, `map`, `multiset`, `multimap`）通常用于快速查找、键值对管理以及保持元素有序，而序列容器（如 `vector`, `deque`, `list`）则更侧重于元素的顺序存储和访问。

因此， 关联容器提供了多种查找函数来支持这一需求：

1. **`find`**：快速定位特定键（或值）的元素。
2. **`count`**：在允许重复键的容器中，计算某个键出现的次数。
3. **`lower_bound`**、**`upper_bound`**：利用容器元素的有序性，找到满足特定条件的边界位置。
4. **`equal_range`**：返回一个范围，其中包括所有等于给定键的元素。

这些函数的存在是为了利用关联容器的内部结构优势（例如，红黑树的有序性和哈希表的快速访问能力），使得关键字相关的查找操作更为高效。





##### 容器比较逻辑：Observers

Observers 是提供有关容器比较逻辑的信息的函数，这在关联容器中尤为重要，因为元素的存储和检索依赖于键（或值）之间的比较结果：

1. **`key_comp`**：返回用于键的比较的函数对象。这对于理解容器如何排序元素，以及如何应用 `lower_bound` 等函数至关重要。
2. **`value_comp`**：在 `set` 和 `multiset` 中，这通常与 `key_comp` 相同，因为每个元素即是一个值也是一个键。









##### 为什么序列容器没有这些函数

序列容器的设计侧重于按顺序存储和管理元素，而不是进行快速查找或键值对管理。例如，`vector` 和 `deque` 提供随机访问能力，但他们不自动维护任何元素的排序状态，除非显式进行排序操作。这意味着：

- **没有内置的快速查找功能**：因为元素不是自动排序的，普通的查找操作需要遍历整个容器，因此复杂度是 O(n)。
- **不需要比较函数**：由于不维护元素的有序状态，不需要像 `key_comp` 或 `value_comp` 这样的比较观察者。

关联容器和序列容器的这些区别反映了它们的设计目的和使用场景的不同。关联容器适用于需要快速基于键的查找、经常进行键相关操作的场景，而序列容器则适合于需要快速顺序访问和简单的元素累加或修改操作的场景。这些设计差异导致了它们各自支持的操作集合的不同。





##### `set`如何保证元素唯一性

**使用比较函数判断相等**：在 C++ 中，如果比较函数 `comp` 用于 `set` 容器，两个元素 `a` 和 `b` 被认为是相等的，当且仅当 `comp(a, b) == false` 并且 `comp(b, a) == false`。这意味着，`a` 不小于 `b` 也不大于 `b`，因此两者相等。

- 当尝试插入一个新元素时，`set` 首先使用比较函数查找正确的插入位置。
- 在查找过程中，如果找到一个已存在的元素 `x`，使得对于待插入元素 `e` 有 `comp(e, x) == false` 且 `comp(x, e) == false`，则认为 `e` 已存在于集合中。
- 如果这样的元素 `x` 被找到，新元素 `e` 不会被插入到 `set` 中，因为 `set` 要保持元素的唯一性。
- 如果没有找到这样的元素，`e` 将被插入到适当的位置以维持元素的顺序。









### 3.1 `set`

#### set数据结构原理

Set 在 C++ 中通常实现为平衡二叉树（最常见的实现为红黑树）。这种数据结构被选择用来实现 set 的原因是它支持高效的搜索、插入和删除操作（均为 O(log n) 复杂度），这是因为红黑树保持了树的平衡，从而保证了在最坏情况下这些操作的效率。

```c++
template<
		class key, 
		class Compare = std::less<key>, 
		class Allocator = std::allocator<key>
> class set;            
```

**`set`**：集合，包含**键**的**有序集**，==键唯一==。









#### 构造set

```c++
// 默认构造
std::set<std::string> a;
a.insert("cat");
a.insert("dog");
a.insert("horse");
// cat dog horse

```

```c++
// 范围构造 C++14
std::set<std::string> b(a.find("dog"), a.end());
// dog horse
```

```c++
// copy construct
std::set<std::string> c(a);
c.insert("another horse");
// another horse cat dog horse
```

```c++
// Move constructor
std::set<std::string> d(std::move(a));
// d: cat dog horse
// a: 空的
```

```c++
// 万能构造
std::set<std::string> e{"one", "two", "three", "five", "eight"};
```

```c++
// 自定义比较
// 这里代表了set是一个无重复（唯一键）的集合，这个键的判定与Pointcmp一致，而非键本身。在以下的例子中，即为不是Point本身，而是PointCmp(Point)
struct Point { double x, y; };
 
struct PointCmp
{
    bool operator()(const Point& lhs, const Point& rhs) const
    {
        //hypot相当于\sqrt(x^2,y^2)
        return std::hypot(lhs.x, lhs.y) < std::hypot(rhs.x, rhs.y);
    }
};
std::set<Point, PointCmp> z = {{2, 5}, {3, 4}, {1, 1}};
z.insert({1, -1}); // This fails because the magnitude of (1,-1) equals (1,1)

/*
当尝试插入 {1, -1} 时，尽管它在平面坐标上与 {1, 1} 不同，但由于它们到原点的距离相同（根据 PointCmp 的定义），set 认为这个点已经存在，因此插入操作失败。
*/
```









#### 查

```c++
std::set<int> s = {1, 2, 3, 4, 5};
```



- `empty()`：检查容器是否为空，复杂度 O(1)。
- `size()`：返回容器中的元素数量，复杂度 O(1)。
- `max_size()`：返回容器可能包含的最大元素数，复杂度 O(1)。





##### Look up

- `count(const Key& key)`：返回某个值的元素数量（对于 set 而言，结果为 0 或 1），复杂度 O(log n)。

- `find(const Key& key)`：查找键值为 key 的元素，复杂度 O(log n)。

  ```c++
  std::cout << "Find 4: " << (s.find(4) != s.end() ? "Found" : "Not Found") << '\n';
  ```

- `contains(const Key& key)`（C++20）：检查键值为 key 的元素是否存在，复杂度 O(log n)。

  ```c++
  // 作用与s.find(5) != s.end()相同，但是在c++20中引入这种表示更清晰
  std::cout << "Contains 5: " << (s.contains(5) ? "Yes" : "No") << '\n';
  ```

- `equal_range(const Key& key)`：返回一个范围，包括所有键等于 key 的元素，复杂度 O(log n)。

  ```c++
  // return std::pair<iterator first, iterator second> [first,second)
  // first指向第一个不小于给定键值的元素
  // second指向第一个大于给定键值的元素
  // 如果没有找到，那么first == second，通知只想给定键应该插入的位置
  ```

- `lower_bound(const Key& key)`：返回指向第一个不小于 key 的元素的迭代器，复杂度 O(log n)。

- `upper_bound(const Key& key)`：返回指向第一个大于 key 的元素的迭代器，复杂度 O(log n)。

  ```c++
  std::cout << "Lower bound for 3: " << *s.lower_bound(3) << '\n';
  std::cout << "Upper bound for 3: " << *s.upper_bound(3) << '\n';
  ```

  



##### Observers

- `key_comp()`：返回用于元素比较的键比较函数，使用两个元素使用与某个实例相同的比较方法来比较。

  ```c++
  std::set<int> s = {10, 20, 30, 40, 50};
  // 获取比较器
  auto comp = s.key_comp();
  int value = 35;
  // 使用比较器
  bool before_end = comp(value, *s.rbegin()); // 比较 value 和 set 中的最大元素
  ```

- `value_comp()`：返回一个函数用来比较value，如果是map的话有用，由于在 set 中值即键，这通常与 `key_comp()` 相同。



#### 改

- `clear()`：清除所有元素，复杂度 O(n)。

- 插入元素，删除元素

  - `insert()`：插入元素，复杂度平均 O(log n)，最坏 O(log n)。

    ```c++
    // 单一元素插入
    // std::pair<iterator,bool> insert(const value_type& val);
    auto res = s.insert(10);
    std::cout << "Inserted 10, success = " << res.second << std::endl;// success
    res = s.insert(10);
    std::cout << "Inserted 10 again, success = " << res.second << std::endl;// false
    
    ```

    ```c++
    // 范围插入
    std::vector<int> vec = {1, 2, 3};
    s.insert(vec.begin(), vec.end());
    // 初始化列表插入
    s.insert({4, 5, 6});
    ```

    

  - `emplace(Args&&... args)`：原地构造元素以插入，复杂度平均 O(log n)，最坏 O(log n)。

    ```c++
    template <class... Args>
    std::pair<iterator,bool> emplace(Args&&... args);
    ```

  - `emplace_hint(iterator pos, Args&&... args)`：在 pos 提示的位置原地构造元素以插入，复杂度通常比无提示的插入要快。

    ```c++
    // 考虑集合的插入不需要给定pos，因为是自主有排序的，但是给定要给pos提示能够帮助更快的插入
    // 通常可以与equal_range联用
    auto hint = s.begin();  // 提供一个合理的插入提示
    s.emplace_hint(hint, 15);  // 在hint位置原地构造元素以插入
    ```

  - `erase(iterator pos)`：删除 pos 指定的元素，复杂度 O(log n)。

    ```c++
    std::set<int> s = {1, 2, 3, 4, 5};
    // 删除单个元素
    // iterator erase(const_iterator position); 返回被删除元素之后的元素的iterator
    s.erase(s.find(3));
    
    // 通过key直接删除
    // size_type erase(const key_type& k); 返回删除元素的数量
    int num_erased = s.erase(1);
    
    // 通过iterator范围删除多个元素
    // iterator erase(const_iterator first, const_iterator last);
    s.erase(s.begin(), s.end());
    
    ```

    > 虽然在关联容器中erase支持直接删除单个元素，但是推荐使用迭代器删除，这样保证了使用的统一性

- `swap(set& other)`：与另一个 set 交换内容，复杂度 O(1)。

- `extract(const iterator& pos)`（C++17）：`extract` 方法从 `set` 中移除一个元素，但与 `erase` 不同，它不直接销毁这个元素，而是返回一个节点句柄（`node_type`），通过这个节点句柄，该元素可以被重新插入到同一类型的另一个 `set` 中，无需复制或移动数据。这可以显著减少操作开销，尤其是对于包含非平凡非廉价复制的数据类型的容器。

  ```c++
   std::set<int> set1 = {1, 2, 3, 4, 5};
  
  // 提取元素 '3'，使用 key 版本
  auto node = set1.extract(3);
  if (!node.empty()) {
      std::cout << "Extracted: " << *node.value() << std::endl;
  }
  
  std::set<int> set2;
  // 将提取的节点插入另一个 set
  set2.insert(std::move(node));
  ```

  

- `merge(std::set& source)`（C++17）：`merge` 方法将一个 `std::set` 中的所有元素转移到另一个 `std::set` 中。这个操作保证只有合法的元素（即不会导致任何键冲突的元素）才会被转移。这是通过将元素的节点直接从一个容器移到另一个容器来完成的，避免了元素的复制和移动，从而提高效率。

  ```c++
  std::set<int> set1 = {1, 2, 3};
  std::set<int> set2 = {3, 4, 5};  // 包含与 set1 重叠的元素
  
  // 合并 set2 到 set1
  set1.merge(set2);
  
  // 合并后，set1 = {1,2,3,4,5}
  // 合并后, set2 = {3}
  ```

  



























### 3.2 `multi-set`

#### `multi-set`数据结构原理

`multiset` 是一种允许重复元素的集合，其主要应用场景是当你需要存储一个元素集合，而这些元素可以重复，并且需要保持元素有序时使用。与 `set` 相比，`multiset` 提供了更灵活的数据结构来处理频繁出现的值。

`multiset` 内部通常使用一种平衡二叉搜索树实现，最常用的是红黑树。这种数据结构允许每个节点存储一个与其他节点的键值相等的数据元素。因此，重复的元素在树中各占一个节点。

当你向 `multiset` 添加一个元素时，容器会使用其内部的比较函数（默认为 `std::less<T>`）来确定新元素应该插入的位置。由于 `multiset` 允许重复，新元素即使与已存在的元素值相同，也会被插入到树中。如果有多个相同的元素，新元素通常会被插入到最后一个相同元素的后面。



Multi-set常用于需要有序和重复元素集合的场景，如统计数据中的频率、数据流的中值查找等。



#### 构造`multi-set`

1. **默认构造**

```c++
std::multiset<int> ms1; // 默认构造，空的multiset
```

2. **初始化列表构造**

   使用初始化列表构造 `multiset`，列表中的元素可以有重复，且会自动排序。

```c++
std::multiset<int> ms2{1, 2, 2, 3, 4}; // 列表初始化，包含重复元素
```

3. **自定义比较器**

```c++
std::multiset<int, std::greater<int>> ms3(ms2.begin(), ms2.end()); // 使用自定义比较器，不然默认是std::less<int>

//自定义比较器
auto comp = [](int a, int b) { return a > b; };
std::multiset<int, decltype(comp)> ms6(comp);  // 创建一个以降序排序的 multiset

```

4. **范围构造**

   使用两个迭代器（开始和结束）来构造 multiset，可以从另一个容器或 multiset 中复制元素。

```c++
std::vector<int> vec = {2, 4, 4, 6};
std::multiset<int> ms3(vec.begin(), vec.end());
```

5. **拷贝构造**

   使用另一个 `multiset` 的副本来创建一个新的 `multiset`。

```c++
std::multiset<int> ms4 = ms2;  // 从 ms2 拷贝元素到 ms4
```

6. **移动构造**

```c++
std::multiset<int> ms5 = std::move(ms4);  // 移动 ms4 的内容到 ms5
```

7. **带有自定义分配器的构造方法**： 

   使用自定义的内存分配器来构造 `multiset`。这对于优化内存使用或使用特定内存池非常有用。

   ```c++
   std::allocator<int> alloc;
   std::multiset<int, std::less<int>, std::allocator<int>> ms7(alloc);
   ```

















#### 增

- **`insert()`**: `insert()` 方法可以用来向 `multiset` 中插入单个元素或一个元素的区间。该操作的时间复杂度通常为 O(log n)，其中 n 是 `multiset` 中元素的数量。

  - 插入单个元素

    ```c++
    std::multiset<int> ms;
    ms.insert(10);
    ms.insert(5);
    ms.insert(15);
    ```

  - 插入(另一个container的）区间

    ```c++
    std::vector<int> values = {20, 40, 30};// container不限，只要这个的迭代器满足输入迭代器的要求
    std::multiset<int> ms;
    ms.insert(values.begin(), values.end());
    ```

    

- **`emplace()`**: 直接在集合中构造元素，避免复制或移动操作。复杂度通常为O(log n)。`emplace` 方法确实$\color{red}{不支持区间构造}$。`emplace` 方法设计的目的是在容器中的指定位置直接构造一个元素，这样可以避免额外的复制或移动操作，从而提高效率。它只能一次构造一个元素，而不是从一个区间构造多个元素。

  - `emplace()` 方法类似于 `insert()`，但它可以直接在 `multiset` 中的正确位置构造元素，避免了复制或移动操作。这可以提高性能，尤其是对于复杂对象的插入。复杂度同样是 O(log n)。

    ```c++
    std::multiset<std::pair<int, std::string>> ms;
    ms.emplace(1, "apple");
    ms.emplace(2, "banana");
    ```

    

- **`emplace_hint`**: 提供一个提示位置，可能提高插入效率。复杂度为O(log n)或更好。

  - `emplace_hint()` 方法接受一个迭代器作为“提示”位置，这个位置是新元素可能插入的位置。如果提示正确，它可以提高插入效率，因为它减少了部分查找过程。复杂度在最优情况下可以达到 O(1)，但如果提示位置不正确，复杂度仍然是 O(log n)。

    ```c++
    std::multiset<int> ms;
    auto it = ms.emplace(10);  // emplace 返回新元素的迭代器
    ms.emplace_hint(it, 9);    // 插入一个小于10的值，提示位置很接近实际位置
    ms.emplace_hint(it, 11);   // 插入一个大于10的值，提示位置依然有用
    ```



















#### 查

- `count()`: 返回特定元素的数量。复杂度为O(log n)。
- `find()`: 查找特定元素。复杂度为O(log n)。
- `contains()`: 检查元素是否存在（C++20）。复杂度为O(log n)。
- `equal_range()`: 返回特定元素的范围（起止迭代器）。复杂度为O(log n)。
- `lower_bound()`, `upper_bound()`: 返回大于或等于、大于某值的第一个位置。复杂度为O(log n)。
  - `auto it = ms.lower_bound(a)`则`*it >= a`
  - `auto it = ms.upper_bound(b)`则`*it > b`

```c++
#include <iostream>
#include <set>

int main() {
    std::multiset<int> ms = {1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5};
    
    // 使用 count() 方法
    int count_3 = ms.count(3);
    std::cout << "Number of '3's: " << count_3 << std::endl;
    
    // 使用 find() 方法
    auto it_find = ms.find(4);
    if (it_find != ms.end()) {
        std::cout << "Found '4' in the multiset." << std::endl;
    } else {
        std::cout << "'4' not found in the multiset." << std::endl;
    }
    
    // 使用 contains() 方法（C++20）
    bool contains_2 = ms.contains(2);
    std::cout << "Multiset contains '2': " << (contains_2 ? "Yes" : "No") << std::endl;

    // 使用 equal_range() 方法
    auto [it_begin, it_end] = ms.equal_range(5);
    std::cout << "Elements equal to '5': ";
    for (auto it = it_begin; it != it_end; ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
    
    // 使用 lower_bound() 和 upper_bound() 方法
    auto it_lower = ms.lower_bound(3);
    auto it_upper = ms.upper_bound(3);
    std::cout << "Elements >= '3' start from: " << *it_lower << std::endl;
    std::cout << "First element > '3' is: " << *it_upper << std::endl;

    return 0;
}

```







#### 删

- `clear()`: 

  - `clear()` 方法用于移除 `multiset` 中的所有元素，使容器变为空。这个操作的复杂度为 O(n)，其中 n 是容器中元素的数量。

    ```c++
    std::multiset<int> ms = {1, 2, 2, 3, 4};
    ms.clear();  // 清空 multiset
    ```

    

- `erase()`: 

  - `erase()` 方法可以删除单个元素或一个元素的区间。删除单个元素的复杂度是 O(log n)，而删除一个区间的复杂度是 O(log n + k)，其中 k 是被删除的元素数。

  - **删除单个元素**：

    当使用键值调用 `erase()` 时（例如 `erase(2)`），它会返回被删除的元素数量。而传入迭代器时（例如 `erase(it)`），没有返回值。

    ```c++
    std::multiset<int> ms = {1, 2, 2, 3, 4};
    auto it = ms.find(2);
    if (it != ms.end()) {
        ms.erase(it);  // 删除一个元素 2
    }
    
    ```

  - **删除一个区间**：

    ```c++
    auto start = ms.lower_bound(2);
    auto end = ms.upper_bound(3);
    ms.erase(start, end);  // 删除所有 2 和 3，erase[start,end)
    ```

    

- `extract()`: 移除节点，不破坏迭代器（C++17）。复杂度为O(log n)。

  ```c++
  std::multiset<int> ms = {1, 2, 2, 3};
  auto node = ms.extract(2);  // 提取一个 2 的节点
  // node 现在包含值 2，可以修改或重新使用
  std::multiset<int> other_ms;
  other_ms.insert(std::move(node));  // 将提取的节点移动到另一个 multiset
  ```

  

- `merge()`: 将另一个multiset中的所有元素合并进来（C++17）。复杂度为O(n + m)。

  ```c++
  std::multiset<int> ms1 = {1, 2, 2};
  std::multiset<int> ms2 = {2, 3, 4};
  ms1.merge(ms2);  // 将 ms2 的元素合并到 ms1 中
  
  // 注意，ms2 现在为空，因为它的所有元素都被移动到了 ms1
  ```

  



> `erase()` 方法从容器中删除一个或多个元素，并释放与这些元素关联的内存。在删除元素时，容器可能需要进行内部结构调整以维持其性能或性质（如平衡二叉树的平衡）。这种调整可能涉及到以下操作：
>
> 1. **内存释放**：被删除元素的内存被释放。任何指向该元素的迭代器、指针或引用都将失效。
> 2. **元素移动**：在某些容器（如 `std::vector` 或 `std::deque`）中，删除一个元素后可能需要移动后续元素来填补空出的位置，这将导致指向移动元素的迭代器、指针或引用失效。
> 3. **结构调整**：对于基于树的容器（如 `std::set` 或 `std::map`），删除节点可能需要额外的节点旋转和重新链接操作来维持树的平衡，这也可能导致除被删除节点外的其他节点的迭代器失效。
>
> `extract()` 方法从 `std::set` 或 `std::multiset` 等基于树的容器中移除元素时，==并不立即重新平衡树==。这一点与 `erase()` 方法不同，后者在删除节点后通常会进行必要的树平衡操作来保证树的性能特性。
>
> 当 `extract()` 方法被调用时，它仅仅将指定的节点从树中“拿出”，而不销毁节点或触及其内容。这个节点和其内容随后被封装在一个 `node_type` 对象中返回。这样做有几个直接的结果：
>
> 1. **树的结构变动**：节点被移除时，树的结构会相应地调整，例如重新链接父节点和子节点，但不会进行广泛的平衡操作。
> 2. **不影响迭代器**：除了指向被提取元素的迭代器外，其他迭代器仍然有效。这是因为树的大部分结构仍保持不变。
> 3. **后续操作可选**：一旦节点被提取，==它可以被修改后重新插入到同一个或不同的容器中==，或者可以被完全丢弃。==如果重新插入，这时候容器将执行必要的平衡操作以维持其性能特性。==
>
> `extract()` 的设计初衷是为了提供一种灵活的操作方式，使开发者能够对容器的元素进行更精细的控制，例如在不同容器之间移动节点而不需要重复构造和销毁。==如果 `extract()` 在移除节点时进行了平衡，那么每次提取和重新插入操作都可能触发一次成本较高的树重平衡，这会降低效率。==













#### 改

- `swap()`: 交换两个集合的内容。复杂度为O(1)。















#### 比较器



* `value_comp()`

  `value_comp()` 方法返回一个可以用来比较 `multiset` 中元素的比较器对象。这个比较器定义了 `multiset` 中元素的排序规则。使用这个比较器，可以确定插入到 `multiset` 的新元素应该位于何处，以及如何维持元素的排序顺序。

* `key_comp()`

  `key_comp()` 方法返回的也是一个比较器，它用于比较 `multiset` 中的键（在这种情况下，键即值）。由于 `multiset` 中的键和值是相同的，`key_comp()` 实际上和 `value_comp()` 返回的比较器是相同的。

```c++
 std::multiset<int, std::greater<int>> ms = {5, 2, 9, 1, 9, 4};

// 获取比较器
auto comp = ms.value_comp();

std::cout << "Elements in the set: ";
for (const auto& e : ms) {
    std::cout << e << " ";
}
std::cout << std::endl;

// 使用比较器手动比较两个元素
int a = 5, b = 9;
if (comp(a, b)) {
    std::cout << "According to the comparator, " << a << " is less than " << b << std::endl;
} else {
    std::cout << "According to the comparator, " << a << " is not less than " << b << std::endl;
}

// 检查 value_comp 和 key_comp 是否相同
auto compKey = ms.key_comp();
if (comp(5, 2) == compKey(5, 2)) {
    std::cout << "value_comp() and key_comp() are equivalent." << std::endl;
}

```





















### 3.3 `map`

#### `map`数据结构原理

`std::map` 是一种关联容器，它存储键值对并按键进行排序。主要解决的问题是如何有效地存储和检索数据，同时保持键的顺序。它允许快速查找、插入和删除操作，具有高效的数据访问能力。

`std::map` 默认使用 `std::less` 来排序其==键==。可以通过提供自定义比较函数来改变排序行为。

标准库提供的 `std::allocator` 是最常用的分配器，它适用于大多数情况。自定义分配器可以用于优化特定类型的内存管理，例如使用内存池。

`std::map` 通常由红黑树实现，这是一种自平衡二叉搜索树。红黑树保证了最坏情况下的关键操作（如搜索、插入、删除）的时间复杂度为 O(log n)。

> 为什么不使用哈希表，是因为红黑树稳定的能够提供O(logn)的查询，而哈希表最差会到O(n)，平均O(1)。

注意与unordered_map的区分，因为需要排序，因此它是使用红黑树对key排序的，因此其任意key查询value的复杂度是logn，而使用哈希表的unordered_map是O(1)；

此外，map和set其实很相似

首先，不能看成map<int,int> == set<pair<int,int>>，因为pair<int,int>只要pair.first,pair.second任意一个不同，就不算一个个体。因此，准确来说，map和set的关系其实是如下

```c++
struct node{
    int key;
    int value;
    
    bool operator==(const node& other) const{
        return this->key == other.key;
    }
    
    bool operator<(const node& other) const{
        return this->key < other.key;
    }
}

// map<int,int> 等同于set<node>，但是官方给你重载了很多方便的符号
```

但是，其实你直接读取的话，还是读出来一个pair的。你仍然需要.first,.second来访问，也就是说它使用的其实是

```c++
struct pair{
    int first,second;
     bool operator==(const node& other) const{
        return this->first == other.first;
    }
    
    bool operator<(const node& other) const{
        return this->second < other.second;
    }
}
```





#### 构造`map`

```c++
#include <map>
std::map<int, std::string> map1; // 默认构造函数
std::map<int, std::string> map2 {{1, "one"}, {2, "two"}}; // 初始化列表构造
std::map<int, std::string> map3(map2.begin(), map2.end()); // 范围构造
std::map<int, std::string> map4(map3); // 拷贝构造
std::map<int, std::string> map5(std::move(map4)); // 移动构造
std::map<int, std::string> map6(std::move(map2), std::allocator<std::pair<const int, std::string>>()); // 分配器与移动构造
```

这里的allocator使用了const的key用来保证红黑树结构稳定，用来排序的key不应能够随意直接改动，只能erase然后insert。



#### 增

- `insert`: 插入键值对，复杂度 O(log n)。

  ```c++
  std::map<int, std::string> myMap;
  
      // 单个键值对的插入
      myMap.insert(std::pair<int, std::string>(1, "one"));
  
      // 使用初始化列表插入单个键值对
      myMap.insert({2, "two"});
  
      // 使用返回值检查插入是否成功
      auto result = myMap.insert({2, "another two"});
      if (!result.second) {
          std::cout << "Element with key 2 not inserted, because it is already in the map." << std::endl;
      }
  
      // 范围插入
      std::map<int, std::string> anotherMap{{3, "three"}, {4, "four"}};
      myMap.insert(anotherMap.begin(), anotherMap.end());
  
      // 打印结果
      for (const auto& pair : myMap) {
          std::cout << pair.first << ": " << pair.second << std::endl;
      }
  ```

  

- `emplace`: 在容器中直接构造键值对，复杂度 O(log n)。

  ```c++
  // 在容器中直接构造键值对
  myMap.emplace(5, "five");
  ```

- `emplace_hint`: 在指定位置前提供一个提示，复杂度通常为 O(log n)，但在最佳情况下接近 O(1)。

  ```c++
  auto it = myMap.begin();
  // 提供一个位置提示进行插入
  myMap.emplace_hint(it, 6, "six");
  ```

- `try_emplace`: C++17 引入，为键插入新元素，如果键已存在，则不进行任何操作，复杂度 O(log n)。

  ```c++
  // 尝试插入新元素，如果键已存在，不会更新现有元素
  myMap.try_emplace(7, "seven");
  myMap.try_emplace(7, "new seven");  // 这条不会执行更新
  ```

  







#### 查

- `empty`: 检查容器是否为空，复杂度 O(1)。

  ```c++
  bool isEmpty = myMap.empty();
  std::cout << "Map is " << (isEmpty ? "empty" : "not empty") << std::endl;
  ```

- `size`: 返回容器中的元素数，复杂度 O(1)。

  ```c++
  std::cout << "Size of map: " << myMap.size() << std::endl;
  ```

- `max_size`: 返回容器可能包含的最大元素数，复杂度 O(1)。

  ```c++
  std::cout << "Max size of map: " << myMap.max_size() << std::endl;
  
  ```

- `find`: 查找键，复杂度 O(log n)。

  ```c++
  auto findIter = myMap.find(5);
  if (findIter != myMap.end()) {
      std::cout << "Found key 5 with value: " << findIter->second << std::endl;
  }
  ```

- `count`: 返回特定键的元素数量（对于 map 总是 1 或 0），复杂度 O(log n)。

  ```c++
  std::cout << "Count of elements with key 5: " << myMap.count(5) << std::endl;
  ```

- `contains`: C++20 引入，检查键是否存在，复杂度 O(log n)。

  ```c++
  if (myMap.contains(5)) {
      std::cout << "Map contains key 5" << std::endl;
  }
  ```

  



**范围查询**

- `equal_range`: 返回特定键的范围，复杂度 O(log n)。

  ```c++
  auto range = myMap.equal_range(5);
  for (auto i = range.first; i != range.second; ++i) {
      std::cout << i->first << ": " << i->second << std::endl;
  }
  ```

- `lower_bound`, `upper_bound`: 分别返回不小于和大于给定键的第一个元素的迭代器，复杂度 O(log n)。

  ```c++
  auto lower = myMap.lower_bound(5);
  auto upper = myMap.upper_bound(5);
  std::cout << "Lower bound of 5: " << lower->second << std::endl;
  if (upper != myMap.end()) {
      std::cout << "Upper bound of 5: " << upper->second << std::endl;
  }
  ```

  

#### 删

- `clear`: 清除所有元素，复杂度 O(n)。

  ```c++
  myMap.clear();
  ```

- `erase`: 删除一个或多个元素，复杂度 O(log n)。

  ```c++
  // 通过键删除
  myMap.erase(5);
  
  // 通过迭代器删除
  auto it = myMap.find(6);
  if (it != myMap.end()) {
      myMap.erase(it);
  }
  
  // 通过迭代器范围删除
  auto range = myMap.equal_range(5);
  myMap.erase(range.first,range.second);
  ```

- `extract`: C++17 引入，提取节点，不破坏迭代器，复杂度 O(log n)。

  ```c++
  auto node = myMap.extract(5);
  if (!node.empty()) {
      std::cout << "Extracted node with key " << node.key() << " and value " << node.mapped() << std::endl;
  }
  ```

  

#### 改

- `swap`: 交换两个 map 的内容，复杂度 O(1)。

  ```c++
  std::map<int, std::string> newMap;
  newMap.swap(myMap);
  ```

- `merge`: C++17 引入，合并两个 map，复杂度 O(n + m).

  ```c++
  std::map<int, std::string> otherMap{{8, "eight"}};
  myMap.merge(otherMap);
  ```

  





#### 比较器

- `key_comp`: 返回用于键比较的函数对象。

  ```c++
  auto comp = myMap.key_comp();
  bool before = comp(1, 2);  // 返回 true 如果 1 在 2 之前
  ```

  

- `value_comp`: 返回用于值比较的函数对象。

  ```c++
  auto vcomp = myMap.value_comp();
  bool before = vcomp(*myMap.find(1), *myMap.find(2));
  ```

  























### 3.4 `multi-map`

#### `multimap`数据结构原理

`multimap` 是 C++ 标准库中的一种关联容器，允许存储多个具有相同键值的元素。它是通过平衡二叉树（例如红黑树）实现的，这使得其具有高效的插入、删除和查找操作。



#### 构造方法

**默认构造函数**：创建一个空的 multimap。

```c++
std::multimap<int, std::string> mmap;
```

**范围构造函数**：用指定范围内的元素构造一个 multimap。

```c++
std::vector<std::pair<int, std::string>> vec = { {1, "one"}, {2, "two"}, {3, "three"} };
std::multimap<int, std::string> mmap(vec.begin(), vec.end());
```

**拷贝构造函数**：用另一个 multimap 的拷贝构造一个新的 multimap。

```c++
std::multimap<int, std::string> mmap1 = { {1, "one"}, {2, "two"} };
std::multimap<int, std::string> mmap2(mmap1);
```

**移动构造函数**：用另一个 multimap 的移动构造一个新的 multimap。

```c++
std::multimap<int, std::string> mmap1 = { {1, "one"}, {2, "two"} };
std::multimap<int, std::string> mmap2(std::move(mmap1));
```









#### Capacity

* `empty`：判断 multimap 是否为空。

```c++
bool isEmpty = mmap.empty();
```

* `size`：返回 multimap 中元素的个数。

```c++
size_t size = mmap.size();
```

* `max_size`：返回 multimap 能容纳的最大元素个数。

```c++
size_t maxSize = mmap.max_size();
```



















#### Modifier

**`clear`**：清空 multimap 中的所有元素。

```c++
mmap.clear();
```





**`insert`**：插入元素。有多种重载：

1. 插入单个元素。

   ```c++
   mmap.insert({4, "four"});
   ```

2. 插入多个元素。

   ```c++
   std::vector<std::pair<int, std::string>> vec = { {5, "five"}, {6, "six"} };
   mmap.insert(vec.begin(), vec.end());
   ```





**`emplace`**：在原地构造并插入元素。

```c++
mmap.emplace(7, "seven");
```





**`emplace_hint`**：在给定位置附近原地构造并插入元素。

```c++
mmap.emplace_hint(mmap.begin(), 8, "eight");
```





**`erase`**：删除元素。有多种重载：返回删除的下一个迭代器

1. 按照迭代器删除。

   ```c++
   auto it = mmap.erase(mmap.begin());
   ```

2. 按照键值删除。

   ```c++
   mmap.erase(1);
   ```





**`swap`**：交换两个 multimap 的内容（交换两个multimap容器）。

```c++
std::multimap<int, std::string> mmap2 = { {9, "nine"}, {10, "ten"} };
mmap.swap(mmap2);
```





**`extract`**：移除并返回元素。（返回被移除的元素）

```c++
auto node = mmap.extract(2);
```





**`merge`**：将另一个 multimap 的元素合并到当前 multimap 中。

```c++
std::multimap<int, std::string> mmap2 = { {11, "eleven"}, {12, "twelve"} };
mmap.merge(mmap2);
```











#### Query

**`count`**：返回与给定键值相等的元素数量。

```c++
size_t count = mmap.count(1);
```

**`find`**：查找具有**给定键值**的元素，返回迭代器。

```c++
auto it = mmap.find(1);
```

**`contains`**：检查 multimap 是否包含具有指定键的元素。

```c++
bool found = mmap.contains(2);
```

**`equal_range`**：返回与给定键值相等的元素范围(`std::pair<std::multimap<int, std::string>::iterator, std::multimap<int, std::string>::iterator>`)

```c++
auto range = mmap.equal_range(1);
```

**`lower_bound`**：返回第一个大于等于给定键值的元素的迭代器。

```c++
auto it = mmap.lower_bound(2);
```

**`upper_bound`**：返回第一个大于给定键值的元素的迭代器。

```c++
auto it = mmap.upper_bound(2);
```













#### Comparison

`key_comp`：返回用于比较键值的比较对象。

```c++
auto comp = mmap.key_comp();
```

`value_comp`：返回用于比较键值对的比较对象。

```c++
auto comp = mmap.value_comp();
```





## 4. 无序关联容器

### 4.0 无序关联容器简介



### 4.1 `unordered_map`







### 4.2 `unordered_multimap`







### 4.3 `unordered_set`





### 4.4 `unordered_multiset`







## 5. Container Adaptor

### 5.0 Adapter简介



### 5.1 `stack`





### 5.2 `queue`





### 5.3 `priority_queue`

**比较函数返回 `true` 时，表示第一个参数的优先级低于第二个参数的优先级**。

下面的比较函数返回如果为true，代表a的优先级低于b的优先级，也就是说这是一个最小堆（因为b的优先级高，所以更靠近堆顶）

```c++
bool compare(T& a, T& b){
    return a > b;
}
```

Priority_queue默认是一个最大堆





## 6. String







## 7. 容器与自定义类型



### 使用Hash表

在 `std::unordered_set` 和 `std::unordered_map` 中，自定义类型需要重载

`std::hash`和`operator==`



* 哈希函数将键对象转换为一个整数值（哈希值），该值用于决定该对象存储在哈希表中的哪个桶（bucket）中。哈希函数应该满足以下特性：

  - **一致性**：对于相同的输入，哈希函数必须始终产生相同的输出。
  - **均匀性**：理想情况下，哈希函数应将输入均匀地分布在所有可能的哈希值上，以减少冲突。

* 当你的自定义类型需要在基于哈希的容器（如 `std::unordered_map` 和 `std::unordered_set`）中使用时，需要重载 `operator==`，以便在键值比较时能够正确判断两个键是否相等。

  当哈希表检测到冲突（即两个不同的键对象具有相同的哈希值）时，需要使用 `==` 运算符来确定这些键是否实际相等。如果两个键的哈希值相同并且 `==` 运算符返回 `true`，则认为它们是相同的键。

```c++
#include <iostream>
#include <unordered_map>
#include <functional>
#include <murmur3.h>  // 假设你有一个 MurmurHash 的实现

// 自定义类型
struct MyKey {
    int a;
    int b;
    std::string c;

    // 重载 == 运算符
    bool operator==(const MyKey& other) const {
        return a == other.a && b == other.b && c == other.c;
    }
};

// 提供自定义哈希函数
namespace std {
    template<>
    struct hash<MyKey> {
        size_t operator()(const MyKey& key) const {
            // 使用 MurmurHash3 来计算哈希值
            size_t hash = 0;
            MurmurHash3_x86_32(&key, sizeof(MyKey), 0, &hash);
            return hash;
        }
    };
}

int main() {
    std::unordered_map<MyKey, std::string> umap;
    umap[{1, 2, "foo"}] = "Hello";
    umap[{3, 4, "bar"}] = "World";

    for (const auto& [key, value] : umap) {
        std::cout << "Key: (" << key.a << ", " << key.b << ", " << key.c << ") => " << value << std::endl;
    }

    return 0;
}

```















### 使用有序容器

#### 重载<号

`std::map` 和 `std::set` 是基于红黑树实现的有序容器。要在这些容器中使用自定义类型作为键，需要重载 `<` 运算符。

```c++
#include <iostream>
#include <map>
#include <set>

// 自定义类型
struct MyKey {
    int a;
    int b;

    bool operator<(const MyKey& other) const {
        return std::tie(a, b) < std::tie(other.a, other.b);
    }
};

int main() {
    std::map<MyKey, std::string> omap;
    omap[{1, 2}] = "Hello";
    omap[{3, 4}] = "World";

    for (const auto& [key, value] : omap) {
        std::cout << "Key: (" << key.a << ", " << key.b << ") => " << value << std::endl;
    }

    std::set<MyKey> oset;
    oset.insert({1, 2});
    oset.insert({3, 4});

    for (const auto& key : oset) {
        std::cout << "Key: (" << key.a << ", " << key.b << ")" << std::endl;
    }

    return 0;
}
```







#### 使用**自定义比较器**

```c++
#include <iostream>
#include <map>
#include <set>
#include <queue>
#include <functional>

// 自定义类型
struct MyKey {
    int a;
    int b;
};

// 自定义比较器
struct MyComparator {
    bool operator()(const MyKey& lhs, const MyKey& rhs) const {
        return std::tie(lhs.a, lhs.b) < std::tie(rhs.a, rhs.b);
    }
};

int main() {
    std::set<MyKey, MyComparator> mySet;
    mySet.insert({2, 1});
    mySet.insert({1, 2});

    for (const auto& key : mySet) {
        std::cout << "Key: (" << key.a << ", " << key.b << ")" << std::endl;
    }

    return 0;
}
```











## 使用容器的几个要素

* 使用自定义比较器增加灵活度

  ```c++
  struct MyComparator {
      bool operator()(const MyKey& lhs, const MyKey& rhs) const {
          return std::tie(lhs.a, lhs.b) < std::tie(rhs.a, rhs.b);
      }
  };
  
  std::set<MyKey, MyComparator> mySet;
  ```

  

* 使用emplace而非insert

* 使用std::move

* 使用std::forward（泛型编程中使用，完美转发）

* 使用智能指针
