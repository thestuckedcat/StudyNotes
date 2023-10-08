给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

例如，121 是回文，而 123 不是。


示例 1：

输入：x = 121
输出：true
示例 2：

输入：x = -121
输出：false
解释：从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
示例 3：

输入：x = 10
输出：false
解释：从右向左读, 为 01 。因此它不是一个回文数。


提示：

-231 <= x <= 231 - 1

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/palindrome-number
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。



# 1.极简

```c++
class Solution {
public:
    bool isPalindrome(int x) {
        string s = to_string(x);
        return s == string(s.rbegin(),s.rend());
    }
};
```



# 2.回文一半

```c++
class Solution {
public:
    bool isPalindrome(int x) {
        int reversed_number = 0;
        
        if(x < 0 || (x % 10 == 0 && x != 0))
        {
            return false;
        }else{
            while(x > reversed_number){
                reversed_number = reversed_number*10 + x % 10;
                x = x / 10;
                
            }
            
            return x == reversed_number || x == reversed_number / 10;
        }
    }
};
```

回转一半，注意要讨论10,100,1000这种情况，实际上所有能被10整除的数都不是回文数，应该加上特判