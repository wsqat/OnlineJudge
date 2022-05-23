其中两处 ... 表示的更新窗⼝数据的地⽅，到时候你直接往⾥⾯填就⾏了。
⽽且，这两个 ... 处的操作分别是扩⼤和缩⼩窗⼝的更新操作，等会你会发现它们操作是完全对称的。



```
/* 滑动窗⼝算法框架 */
void slidingWindow(string s, string t) {
 unordered_map<char, int> need, window;
 for (char c : t) need[c]++;
 
 int left = 0, right = 0;
 int valid = 0; 
 while (right < s.size()) {
 // c 是将移⼊窗⼝的字符
 char c = s[right];
 // 增⼤窗⼝
 right++;
 // 进⾏窗⼝内数据的⼀系列更新
 ...
 /*** debug 输出的位置 ***/
 printf("window: [%d, %d)\n", left, right);
 /********************/
 
 // 判断左侧窗⼝是否要收缩
 while (window needs shrink) {
 // d 是将移出窗⼝的字符
 char d = s[left];
 // 缩⼩窗⼝
 left++;
 // 进⾏窗⼝内数据的⼀系列更新
 ...
 }
 }
}
```

滑动窗⼝算法的思路是这样：
```
1、我们在字符串 S 中使⽤双指针中的左右指针技巧，初始化 left = right = 0，把索引左闭右开区间
[left, right) 称为⼀个「窗⼝」。
PS：理论上你可以设计两端都开或者两端都闭的区间，但设计为左闭右开区间是最⽅便处理的。因为
这样初始化 left = right = 0 时区间 [0, 0) 中没有元素，但只要让 right 向右移动（扩⼤）⼀
位，区间 [0, 1) 就包含⼀个元素 0 了。如果你设置为两端都开的区间，那么让 right 向右移动⼀
位后开区间 (0, 1) 仍然没有元素；如果你设置为两端都闭的区间，那么初始区间 [0, 0] 就包含了
⼀个元素。这两种情况都会给边界处理带来不必要的麻烦。
2、我们先不断地增加 right 指针扩⼤窗⼝ [left, right)，直到窗⼝中的字符串符合要求（包含了 T 中
的所有字符）。
3、此时，我们停⽌增加 right，转⽽不断增加 left 指针缩⼩窗⼝ [left, right)，直到窗⼝中的字符串
不再符合要求（不包含 T 中的所有字符了）。同时，每次增加 left，我们都要更新⼀轮结果。
4、重复第 2 和第 3 步，直到 right 到达字符串 S 的尽头。
这个思路其实也不难，第 2 步相当于在寻找⼀个「可⾏解」，然后第 3 步在优化这个「可⾏解」，最终找到
最优解，也就是最短的覆盖⼦串。左右指针轮流前进，窗⼝⼤⼩增增减减，窗⼝不断向右滑动，这就是「滑
动窗⼝」这个名字的来历。

```

⾸先，初始化 window 和 need 两个哈希表，记录窗⼝中的字符和需要凑⻬的字符：
unordered_map<char, int> need, window;
for (char c : t) need[c]++;
然后，使⽤ left 和 right 变量初始化窗⼝的两端，不要忘了，区间 [left, right) 是左闭右开的，所以
初始情况下窗⼝没有包含任何元素：
int left = 0, right = 0;
int valid = 0; 
while (right < s.size()) {
 // 开始滑动
}
其中 valid 变量表示窗⼝中满⾜ need 条件的字符个数，如果 valid 和 need.size 的⼤⼩相同，则说明窗
⼝已满⾜条件，已经完全覆盖了串 T。
现在开始套模板，只需要思考以下四个问题：
1、当移动 right 扩⼤窗⼝，即加⼊字符时，应该更新哪些数据？
2、什么条件下，窗⼝应该暂停扩⼤，开始移动 left 缩⼩窗⼝？
3、当移动 left 缩⼩窗⼝，即移出字符时，应该更新哪些数据？
4、我们要的结果应该在扩⼤窗⼝时还是缩⼩窗⼝时进⾏更新？


如果⼀个字符进⼊窗⼝，应该增加 window 计数器；如果⼀个字符将移出窗⼝的时候，应该减少 window 计
数器；当 valid 满⾜ need 时应该收缩窗⼝；应该在收缩窗⼝的时候更新最终结果。


> needs 和 window 相当于计数器，分别记录 T 中字符出现次数和「窗⼝」中的相应字符
的出现次数。






## 76. 最小覆盖子串

https://leetcode.cn/problems/minimum-window-substring

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。



注意：

对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
如果 s 中存在这样的子串，我们保证它是唯一的答案。
```

示例 1：

输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
```


### code
```
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> need, window;
        for(char c : t) need[c]++;
        int left=0, right=0;
        int valid=0;
         // 记录最⼩覆盖⼦串的起始索引及⻓度
        int start=0,len=INT_MAX;
        while(right<s.size()){
            // c 是将移⼊窗⼝的字符
            char c = s[right];
             // 扩⼤窗⼝
            right++;
            // 进⾏窗⼝内数据的⼀系列更新
            if(need.count(c)){
                window[c]++;
                if(need[c]==window[c]){
                    valid++;
                }
            }
            
             // 判断左侧窗⼝是否要收缩
            while(need.size()==valid){
                 // 在这⾥更新最⼩覆盖⼦串
                if(right-left<len){
                    start = left;
                    len = right - left;
                }
                
                 // d 是将移出窗⼝的字符
                char d = s[left];
                 // 缩⼩窗⼝
                left++;
                 // 进⾏窗⼝内数据的⼀系列更新
                if(need.count(d)){
                    if(need[d]==window[d]){
                        valid--;
                    }
                    window[d]--;
                }

            }

        }
        
         // 返回最⼩覆盖⼦串
        return len==INT_MAX?"":s.substr(start, len);
    }
};
```

需要注意的是，当我们发现某个字符在 window 的数量满⾜了 need 的需要，就要更新 valid，表示有⼀个
字符已经满⾜要求。⽽且，你能发现，两次对窗⼝内数据的更新操作是完全对称的。
当 valid == need.size() 时，说明 T 中所有字符已经被覆盖，已经得到⼀个可⾏的覆盖⼦串，现在应该
开始收缩窗⼝了，以便得到「最⼩覆盖⼦串」。
移动 left 收缩窗⼝时，窗⼝内的字符都是可⾏解，所以应该在收缩窗⼝的阶段进⾏最⼩覆盖⼦串的更新，
以便从可⾏解中找到⻓度最短的最终结果。


##  567 题「字符串的排列」
https://leetcode.cn/problems/permutation-in-string

给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。

换句话说，s1 的排列之一是 s2 的 子串 。

 

示例 1：

输入：s1 = "ab" s2 = "eidbaooo"
输出：true
解释：s2 包含 s1 的排列之一 ("ba").

```
class Solution {
public:
    bool checkInclusion(string s1, string s2) {

        if (s1.size() > s2.size()) {
            return false;
        }
        unordered_map<char, int> need, window;
        for(char c: s1) need[c]++;
        int left=0, right=0, valid=0;

        while(right<s2.size()){
            char c = s2[right];
            right++;
            if(need.count(c)){
                window[c]++;
                if(window[c]==need[c]){
                    valid++;
                }
            }

            while(right-left>=s1.size()){
                if (valid==need.size()){
                    return true;
                }

                char d = s2[left];
                left++;
                if(need.count(d)){
                    if(window[d]==need[d]){
                        valid--;
                    }
                    window[d]--;
                }
            }
            
        }

        return false;
    }
};

```



## 438. 找到字符串中所有字母异位词


给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。

 

示例 1:

输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/find-all-anagrams-in-a-string
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


    https://leetcode.cn/problems/find-all-anagrams-in-a-string/solution/hua-dong-chuang-kou-tong-yong-si-xiang-jie-jue-zi-/

```
vector<int> findAnagrams(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;
    vector<int> res; // 记录结果
    while (right < s.size()) {
        char c = s[right];
        right++;
        // 进行窗口内数据的一系列更新
        if (need.count(c)) {
            window[c]++;
            if (window[c] == need[c]) 
                valid++;
        }
        // 判断左侧窗口是否要收缩
        while (right - left >= t.size()) {
            // 当窗口符合条件时，把起始索引加入 res
            if (valid == need.size())
                res.push_back(left);
            char d = s[left];
            left++;
            // 进行窗口内数据的一系列更新
            if (need.count(d)) {
                if (window[d] == need[d])
                    valid--;
                window[d]--;
            }
        }
    }
    return res;
}
```


## 3. 无重复字符的最长子串

https://leetcode.cn/problems/longest-substring-without-repeating-characters

给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

 

示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。


```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int> windows;
        int left=0,right=0,valid=0;

        int start=0, len=0;
        while(right<s.size()){
            char c = s[right];
            right++;
            windows[c]++;
            

            while(windows[c]>1){
                
                char d = s[left];
                left++;
                windows[d]--;
                

            }
            len = max(len, right-left);
        }
        return len;
    }
};
```
