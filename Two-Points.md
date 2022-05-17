在处理数组和链表相关问题时，双指针技巧是经常⽤到的，双指针技巧主要分为两类：左右指针和快慢指
针。
所谓左右指针，就是两个指针相向⽽⾏或者相背⽽⾏；⽽所谓快慢指针，就是两个指针同向⽽⾏，⼀快⼀
慢。
对于单链表来说，⼤部分技巧都属于快慢指针，前⽂ 单链表的六⼤解题套路 都涵盖了，⽐如链表环判断，倒
数第 K 个链表节点等问题，它们都是通过⼀个 fast 快指针和⼀个 slow 慢指针配合完成任务。
在数组中并没有真正意义上的指针，但我们可以把索引当做数组中的指针，这样也可以在数组中施展双指针
技巧，本⽂主要讲数组相关的双指针算法。


# ⼀、快慢指针技巧
数组问题中⽐较常⻅且难度不⾼的的快慢指针技巧，是让你原地修改数组。


## 26. 删除有序数组中的重复项

https://leetcode.cn/problems/remove-duplicates-from-sorted-array

给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。

由于在某些语言中不能改变数组的长度，所以必须将结果放在数组nums的第一部分。更规范地说，如果在删除重复项之后有 k 个元素，那么 nums 的前 k 个元素应该保存最终结果。

将最终结果插入 nums 的前 k 个位置后返回 k 。

不要使用额外的空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

判题标准:

系统会用下面的代码来测试你的题解:


### code
```
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int len = nums.size();
        if (len == 0 ) return 0;
        int slow=0, fast=0;
        while(fast<len){
            if(nums[fast]!=nums[slow]){
                slow++;
                nums[slow] = nums[fast];
            }
            fast++;
        }
        return slow+1;
    }
};
```

## 83. 删除排序链表中的重复元素

https://leetcode.cn/problems/remove-duplicates-from-sorted-list/

给定一个已排序的链表的头 head ， 删除所有重复的元素，使每个元素只出现一次 。返回 已排序的链表 

### code
```
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int len = nums.size();
        if (len == 0 ) return 0;
        int slow=0, fast=0;
        while(fast<len){
            if(nums[fast]!=nums[slow]){
                slow++;
                nums[slow] = nums[fast];
            }
            fast++;
        }
        return slow+1;
    }
};
```





## 27. 移除元素
> https://leetcode.cn/problems/remove-element

给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

```
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。
```
### 解题思路

⽬要求我们把 nums 中所有值为 val 的元素原地删除，依然需要使⽤快慢指针技巧：
如果 fast 遇到值为 val 的元素，则直接跳过，否则就赋值给 slow 指针，并让 slow 前进⼀步。


### code
```
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int fast=0, slow=0;
        while(fast<nums.size()){
            if(nums[fast]!=val){
                nums[slow] = nums[fast];
                slow+=1;
            }
            fast+=1;
        }
        return slow;
    }
};
```

## 283. 移动零
https://leetcode.cn/problems/move-zeroes/

给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

请注意 ，必须在不复制数组的情况下原地对数组进行操作。
### 思路
⽬让我们将所有 0 移到最后，其实就相当于移除 nums 中的所有 0，然后再把后⾯的元素都赋值为 0 即
可。

### code
```
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int fast=0, slow=0;
        while(fast<nums.size()){
            if(nums[fast]!=0){
                nums[slow] = nums[fast];
                slow+=1;
            }
            fast+=1;
        }

        for(;slow<nums.size();slow++){
            nums[slow]=0;
        }

    }
};
```

## 167. 两数之和 II - 输入有序数组

https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted

给你一个下标从 1 开始的整数数组 numbers ，该数组已按 非递减顺序排列  ，请你从数组中找出满足相加之和等于目标数 target 的两个数。如果设这两个数分别是 numbers[index1] 和 numbers[index2] ，则 1 <= index1 < index2 <= numbers.length 。

以长度为 2 的整数数组 [index1, index2] 的形式返回这两个整数的下标 index1 和 index2。

你可以假设每个输入 只对应唯一的答案 ，而且你 不可以 重复使用相同的元素。

你所设计的解决方案必须只使用常量级的额外空间。

### 思路
要数组有序，就应该想到双指针技巧。这道题的解法有点类似⼆分查找，通过调节 left 和 right 就可以
调整 sum 的⼤⼩

### code
```
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int left=0, right=numbers.size()-1;
        while(left<right){
            if(numbers[left]+numbers[right]==target){
                return {left+1,right+1};
            }else if(numbers[left]+numbers[right]<target){
                left++;
            }else{
                right--;
            }
        }
        return {-1,-1};
    }
};
```

## 344. 反转字符串
https://leetcode.cn/problems/reverse-string

编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 s 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

### 思路
双指针

### code
```
class Solution {
public:
    void reverseString(vector<char>& s) {
        int left=0,right=s.size()-1;
        while(left<right){
            char c = s[right];
            s[right] = s[left];
            s[left] = c;
            left++;
            right--;
        }
    }
};
```

# 5


```
class Solution {
public:

    string getLongestPalindrome(string s, int l, int r){
        while(l>=0 && r < s.size() && s[l] == s[r]){
            l--;
            r++;
        }
        return s.substr(l+1,r-1);
    }

    string longestPalindrome(string s) {
        string res="";
        for(int i=0;i<s.size();i++){
            string s1 = getLongestPalindrome(s, i, i);
            string s2 = getLongestPalindrome(s, i, i+1);
            res = res.size()>s1.size()?res:s1;
            res = res.size()>s2.size()?res:s2;
        }
        return res;
    }
};  


class Solution {
public:
    pair<int, int> expandAroundCenter(const string& s, int left, int right) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            --left;
            ++right;
        }
        return {left + 1, right - 1};
    }

    string longestPalindrome(string s) {
        int start = 0, end = 0;
        for (int i = 0; i < s.size(); ++i) {
            auto [left1, right1] = expandAroundCenter(s, i, i);
            auto [left2, right2] = expandAroundCenter(s, i, i + 1);
            if (right1 - left1 > end - start) {
                start = left1;
                end = right1;
            }
            if (right2 - left2 > end - start) {
                start = left2;
                end = right2;
            }
        }
        return s.substr(start, end - start + 1);
    }
};

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/longest-palindromic-substring/solution/zui-chang-hui-wen-zi-chuan-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
