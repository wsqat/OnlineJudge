# 一、动态规划

https://labuladong.gitee.io/algo/3/23/66/

首先，虽然动态规划的核心思想就是穷举求最值，但是问题可以千变万化，穷举所有可行解其实并不是一件容易的事，需要你熟练掌握递归思维，只有列出正确的「状态转移方程」，才能正确地穷举。而且，你需要判断算法问题是否具备「最优子结构」，是否能够通过子问题的最值得到原问题的最值。另外，动态规划问题存在「重叠子问题」，如果暴力穷举的话效率会很低，所以需要你使用「备忘录」或者「DP table」来优化穷举过程，避免不必要的计算。

以上提到的重叠子问题、最优子结构、状态转移方程就是动态规划三要素。具体什么意思等会会举例详解，但是在实际的算法问题中，写出状态转移方程是最困难的，这也就是为什么很多朋友觉得动态规划问题困难的原因，我来提供我总结的一个思维框架，辅助你思考状态转移方程：

明确 base case -> 明确「状态」-> 明确「选择」 -> 定义 dp 数组/函数的含义。

```
# 自顶向下递归的动态规划
def dp(状态1, 状态2, ...):
    for 选择 in 所有可能的选择:
        # 此时的状态已经因为做了选择而改变
        result = 求最值(result, dp(状态1, 状态2, ...))
    return result

# 自底向上迭代的动态规划
# 初始化 base case
dp[0][0][...] = base case
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)
```

## 509. 斐波那契数

```
class Solution {
public:
    // 暴力
    int fib_1(int n) {
        if (n==0) return 0;
        if (n==1||n==2) return 1;
        else return fib(n-1)+fib(n-2);
    }

    //dp
    int fib(int n) {
        if (n==0) return 0;
        vector<int> dp(n+1);
        dp[0]=0, dp[1]=1;
        for(int i=2;i<=n;i++){
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n];
    }
};
```

那么，既然知道了这是个动态规划问题，就要思考如何列出正确的状态转移方程？

1、确定 base case，这个很简单，显然目标金额 amount 为 0 时算法返回 0，因为不需要任何硬币就已经凑出目标金额了。

2、确定「状态」，也就是原问题和子问题中会变化的变量。由于硬币数量无限，硬币的面额也是题目给定的，只有目标金额会不断地向 base case 靠近，所以唯一的「状态」就是目标金额 amount。

3、确定「选择」，也就是导致「状态」产生变化的行为。目标金额为什么变化呢，因为你在选择硬币，你每选择一枚硬币，就相当于减少了目标金额。所以说所有硬币的面值，就是你的「选择」。

4、明确 dp 函数/数组的定义。我们这里讲的是自顶向下的解法，所以会有一个递归的 dp 函数，一般来说函数的参数就是状态转移中会变化的量，也就是上面说到的「状态」；函数的返回值就是题目要求我们计算的量。就本题来说，状态只有一个，即「目标金额」，题目要求我们计算凑出目标金额所需的最少硬币数量。

所以我们可以这样定义 dp 函数：dp(n) 表示，输入一个目标金额 n，返回凑出目标金额 n 所需的最少硬币数量。


```
首先判断是不是DP，如何判断？
1. 要符合「最优子结构」，子问题间必须互相独立

是DP，则暴力穷举
2. 找到状态转移方程

如何聪明的暴力穷举？备忘录或者DP table
3. 优化递归树，消除重叠子问题。
- 备忘录：自顶向下
- DP table：自下向上
```



## 322. 零钱兑换

给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。

 

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/coin-change
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


示例 1：

输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1

### 暴力
```
    int coinChange(vector<int>& coins, int amount) {
        return dp(coins, amount);
    }

    int dp(vector<int>& coins, int amount) {
        if (amount==0) return 0;
        if (amount<0) return -1;

        int res = INT_MAX;
        for(int coin: coins){
            int subPro = dp(coins, amount-coin);
            if ( subPro==-1) continue;
            res = min(res, 1+ subPro);
        }
        return res==INT_MAX?-1:res;
    }
```

### 备忘录

https://labuladong.gitee.io/algo/3/23/66/

备忘录初始化为一个不会被取到的特殊值，代表还未被计算

```
    vector<int> memo;
    int coinChange(vector<int>& coins, int amount) {
        memo.resize(amount+1, -666);
        return dp(coins, amount);
    }

    int dp(vector<int>& coins, int amount) {
        if (amount==0) return 0;
        if (amount<0) return -1;
        if (memo[amount]!=-666) return memo[amount];
        int res = INT_MAX;
        for(int coin: coins){
            int subPro = dp(coins, amount-coin);
            if ( subPro==-1) continue;
            res = min(res, 1+ subPro);
        }
        memo[amount] = res==INT_MAX?-1:res;
        return memo[amount];
    }
```



### DP-table

dp 数组的定义：当目标金额为 i 时，至少需要 dp[i] 枚硬币凑出。


```
    int coinChange(vector<int>& coins, int amount) {
        //dp 数组的定义：当目标金额为 i 时，至少需要 dp[i] 枚硬币凑出。
        vector<int> dp(amount+1, amount+1);
        // base case
        dp[0] = 0;
        // 外层 for 循环在遍历所有状态的所有取值
        for(int i=0;i<amount+1;i++){
            // 内层 for 循环在求所有选择的最小值
            for(int coin: coins){
                // 子问题无解，跳过
                if( i<coin) continue;
                dp[i] = min(dp[i], 1+dp[i-coin]);
            }
        }


        return dp[amount]==amount+1?-1:dp[amount];
    }

```

# 二、动态规划设计：最长递增子序列

https://labuladong.gitee.io/algo/3/23/67/


总结一下如何找到动态规划的状态转移关系：

1、明确 dp 数组的定义。这一步对于任何动态规划问题都很重要，如果不得当或者不够清晰，会阻碍之后的步骤。

2、根据 dp 数组的定义，运用数学归纳法的思想，假设 dp[0...i-1] 都已知，想办法求出 dp[i]，一旦这一步完成，整个题目基本就解决了。

但如果无法完成这一步，很可能就是 dp 数组的定义不够恰当，需要重新定义 dp 数组的含义；或者可能是 dp 数组存储的信息还不够，不足以推出下一步的答案，需要把 dp 数组扩大成二维数组甚至三维数组。

目前的解法是标准的动态规划，但对最长递增子序列问题来说，这个解法不是最优的，可能无法通过所有测试用例了，下面讲讲更高效的解法。

## 300. 最长递增子序列

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/longest-increasing-subsequence
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


> nums[i] 为结尾的「最⼤⼦数组和」为 dp[i]
```
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> dp(nums.size(),1);
        for(int i=0;i<nums.size();i++){
            for(int j=0;j<i;j++){
                if (nums[i]>nums[j]){
                    dp[i] = max(dp[i],dp[j]+1);
                }
            }
        }

        int res=0;
        for(int i=0;i<nums.size();i++){
            res = max(res, dp[i]);
        }
        return res;
    }
};
```


### 354. 俄罗斯套娃信封问题


给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

注意：不允许旋转信封。

 
示例 1：

输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/russian-doll-envelopes
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

- max_element函数可以返回容器中的最大值。
- min_element函数可以返回容器中的最小值。
```
函数格式为：
max_element(first,end,cmp)
min_element(first,end,cmp)
```


```
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        if (envelopes.empty()) {
            return 0;
        }
        
        int n = envelopes.size();
        sort(envelopes.begin(), envelopes.end(), [](const auto& e1, const auto& e2) {
            return e1[0] < e2[0] || (e1[0] == e2[0] && e1[1] > e2[1]);
        });

        vector<int> f(n, 1);
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (envelopes[j][1] < envelopes[i][1]) {
                    f[i] = max(f[i], f[j] + 1);
                }
            }
        }
        return *max_element(f.begin(), f.end());
    }

```


- lower_bound()返回值是一个迭代器,返回指向大于等于key的第一个值的位置
- 对应lower_bound()函数是upper_bound()函数，它返回大于等于key的最后一个元素
- 也同样是要求有序数组，若数组中无重复元素，则两者返回值相同

> 基于二分查找的动态规划

https://leetcode.cn/problems/russian-doll-envelopes/solution/e-luo-si-tao-wa-xin-feng-wen-ti-by-leetc-wj68/

```
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        if (envelopes.empty()) {
            return 0;
        }
        
        int n = envelopes.size();
        sort(envelopes.begin(), envelopes.end(), [](const auto& e1, const auto& e2) {
            return e1[0] < e2[0] || (e1[0] == e2[0] && e1[1] > e2[1]);
        });

        vector<int> f = {envelopes[0][1]};
        for (int i = 1; i < n; ++i) {
            if (int num = envelopes[i][1]; num > f.back()) {
                f.push_back(num);
            }
            else {
                auto it = lower_bound(f.begin(), f.end(), num);
                *it = num;
            }
        }
        return f.size();
    }

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/russian-doll-envelopes/solution/e-luo-si-tao-wa-xin-feng-wen-ti-by-leetc-wj68/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


## 3.最优子结构原理和 DP 数组遍历方向

> https://labuladong.gitee.io/algo/3/23/68/

当然这也不是动态规划问题，旨在说明，最优子结构并不是动态规划独有的一种性质，能求最值的问题大部分都具有这个性质；但反过来，最优子结构性质作为动态规划问题的必要条件，一定是让你求最值的，以后碰到那种恶心人的最值题，思路往动态规划想就对了，这就是套路。

动态规划不就是从最简单的 base case 往后推导吗，可以想象成一个链式反应，以小博大。但只有符合最优子结构的问题，才有发生这种链式反应的性质。

找最优子结构的过程，其实就是证明状态转移方程正确性的过程，方程符合最优子结构就可以写暴力解了，写出暴力解就可以看出有没有重叠子问题了，有则优化，无则 OK。这也是套路，经常刷题的读者应该能体会。



## 4.


### 53. 最大子数组和

给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

子数组 是数组中的一个连续部分。

 

示例 1：

输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/maximum-subarray
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

> 以nums[i] 为结尾的「最⼤⼦数组和」为 dp[i]



```
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        //nums[i] 为结尾的「最⼤⼦数组和」为 dp[i]
        //类似寻找最大最小值的题目，初始值一定要定义成理论上的最小最大值

        vector<int> dp;
        dp.resize(nums.size());
        dp[0]=nums[0];
        int res = dp[0];
        for(int i=1;i<nums.size();i++){
            dp[i] = max(nums[i], dp[i-1]+nums[i]); 
            res = max(res, dp[i]);
        }
        return res;
    }
};
```

### 494. 目标和

给你一个整数数组 nums 和一个整数 target 。

向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

 

示例 1：

输入：nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有 5 种方法让最终目标和为 3 。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/target-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


```
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        if (nums.size() == 0) return 0;
        return dp(nums, 0, target);
    }

    unordered_map<string,int> memo;

    int dp(vector<int>nums, int i, int remain){
        if (i==nums.size()){
            if (remain==0) return 1;
            return 0;
        }
        string key = to_string(i)+","+to_string(remain);
        if(memo.count(key)){
            return memo[key];
        }

        int res = dp(nums, i+1, remain-nums[i]) + dp(nums, i+1, remain+nums[i]);
        memo[key] = res;
        return res;
    }
};
```

https://labuladong.gitee.io/algo/3/23/71/
```
其实，这个问题可以转化为一个子集划分问题，而子集划分问题又是一个典型的背包问题。动态规划总是这么玄学，让人摸不着头脑……

首先，如果我们把 nums 划分成两个子集 A 和 B，分别代表分配 + 的数和分配 - 的数，那么他们和 target 存在如下关系：

sum(A) - sum(B) = target
sum(A) = target + sum(B)
sum(A) + sum(A) = target + sum(B) + sum(A)
2 * sum(A) = target + sum(nums)
综上，可以推出 sum(A) = (target + sum(nums)) / 2，也就是把原问题转化成：nums 中存在几个子集 A，使得 A 中元素的和为 (target + sum(nums)) / 2？

类似的子集划分问题我们前文 经典背包问题：子集划分 讲过，现在实现这么一个函数：

/* 计算 nums 中有几个子集的和为 sum */
int subsets(int[] nums, int sum) {}
然后，可以这样调用这个函数：

int findTargetSumWays(int[] nums, int target) {
    int sum = 0;
    for (int n : nums) sum += n;
    // 这两种情况，不可能存在合法的子集划分
    if (sum < Math.abs(target) || (sum + target) % 2 == 1) {
        return 0;
    }
    return subsets(nums, (sum + target) / 2);
}
好的，变成背包问题的标准形式：

有一个背包，容量为 sum，现在给你 N 个物品，第 i 个物品的重量为 nums[i - 1]（注意 1 <= i <= N），每个物品只有一个，请问你有几种不同的方法能够恰好装满这个背包？

现在，这就是一个正宗的动态规划问题了，下面按照我们一直强调的动态规划套路走流程：

第一步要明确两点，「状态」和「选择」。

对于背包问题，这个都是一样的，状态就是「背包的容量」和「可选择的物品」，选择就是「装进背包」或者「不装进背包」。

第二步要明确 dp 数组的定义。

按照背包问题的套路，可以给出如下定义：

dp[i][j] = x 表示，若只在前 i 个物品中选择，若当前背包的容量为 j，则最多有 x 种方法可以恰好装满背包。

翻译成我们探讨的子集问题就是，若只在 nums 的前 i 个元素中选择，若目标和为 j，则最多有 x 种方法划分子集。

根据这个定义，显然 dp[0][..] = 0，因为没有物品的话，根本没办法装背包；但是 dp[0][0] 应该是个例外，因为如果背包的最大载重为 0，「什么都不装」也算是一种装法，即 dp[0][0] = 1。

我们所求的答案就是 dp[N][sum]，即使用所有 N 个物品，有几种方法可以装满容量为 sum 的背包。

第三步，根据「选择」，思考状态转移的逻辑。

回想刚才的 dp 数组含义，可以根据「选择」对 dp[i][j] 得到以下状态转移：

如果不把 nums[i] 算入子集，或者说你不把这第 i 个物品装入背包，那么恰好装满背包的方法数就取决于上一个状态 dp[i-1][j]，继承之前的结果。

如果把 nums[i] 算入子集，或者说你把这第 i 个物品装入了背包，那么只要看前 i - 1 个物品有几种方法可以装满 j - nums[i-1] 的重量就行了，所以取决于状态 dp[i-1][j-nums[i-1]]。

PS：注意我们说的 i 是从 1 开始算的，而数组 nums 的索引时从 0 开始算的，所以 nums[i-1] 代表的是第 i 个物品的重量，j - nums[i-1] 就是背包装入物品 i 之后还剩下的容量。

由于 dp[i][j] 为装满背包的总方法数，所以应该以上两种选择的结果求和，得到状态转移方程：

dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i-1]];
然后，根据状态转移方程写出动态规划算法：
```

定义二维数组 dp，其中 dp[i][j] 表示在数组 nums 的前 i 个数中选取元素，使得这些元素之和等于 j 的方案数。假设数组 nums 的长度为 nn，则最终答案为 dp[n][neg]。

https://leetcode.cn/problems/target-sum/solution/mu-biao-he-by-leetcode-solution-o0cp/

```
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum = 0;
        for (int& num : nums) {
            sum += num;
        }
        int diff = sum - target;
        if (diff < 0 || diff % 2 != 0) {
            return 0;
        }
        int n = nums.size(), neg = diff / 2;
        vector<vector<int>> dp(n + 1, vector<int>(neg + 1));
        dp[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            int num = nums[i - 1];
            for (int j = 0; j <= neg; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= num) {
                    dp[i][j] += dp[i - 1][j - num];
                }
            }
        }
        return dp[n][neg];
    }

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/target-sum/solution/mu-biao-he-by-leetcode-solution-o0cp/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



## 72. 编辑距离


给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符
 

示例 1：

输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')



> 递归

```
class Solution {
public:
    int minDistance(string word1, string word2) {
        int m=word1.size(), n=word2.size();
        return dp(word1, m-1, word2, n-1);
    }

    int dp(string s1, int i, string s2, int j){
        if (i==-1) return j+1;
        if (j==-1) return i+1;

        if(s1[i]==s2[j])
            return dp(s1, i-1, s2, j-1);
        else{
            return Min(
                dp(s1, i-1, s2, j)+1,
                dp(s1, i, s2, j-1)+1,
                dp(s1, i-1, s2, j-1)+1
            );
        }

    }

    int Min(int a, int b, int c){
        return min(a, min(b,c));
    }
};
```


> 备忘录

```
class Solution {
public:
    vector<vector<int>> memo;
    int minDistance(string word1, string word2) {
        int m=word1.size(), n=word2.size();
        vector<vector<int>> matrix(m,vector<int>(n,-1));
        memo = matrix;
        return dp(word1, m-1, word2, n-1);
    }

    int dp(string s1, int i, string s2, int j){
        if (i==-1) return j+1;
        if (j==-1) return i+1;

        if(memo[i][j]!=-1){
            return memo[i][j];
        }

        if(s1[i]==s2[j])
            memo[i][j] = dp(s1, i-1, s2, j-1);
        else{
            memo[i][j] =  Min(
                dp(s1, i-1, s2, j)+1,
                dp(s1, i, s2, j-1)+1,
                dp(s1, i-1, s2, j-1)+1
            );
        }

        return memo[i][j];
    }

    int Min(int a, int b, int c){
        return min(a, min(b,c));
    }
};
```

> dp table

dp(s1, i - 1, s2, j - 1)  // 跳过

dp(s1, i, s2, j - 1) + 1, // 插⼊
dp(s1, i - 1, s2, j) + 1, // 删除
dp(s1, i - 1, s2, j - 1) + 1 // 替换
 
 
```
class Solution {
public:
    
    int minDistance(string word1, string word2) {
        int m=word1.size(), n=word2.size();
        //s1[0..i] 和 s2[0..j] 的最⼩编辑距离是 dp[i+1][j+1]
        vector<vector<int>> dp(m+1,vector<int>(n+1));
        for(int i=1;i<=m;i++){
            dp[i][0] = i;
        }
        for(int i=1;i<=n;i++){
            dp[0][i] = i;
        }

        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(word1[i-1]==word2[j-1]){
                    dp[i][j] = dp[i-1][j-1]; 
                }else{
                    dp[i][j] = Min(dp[i-1][j-1]+1, dp[i][j-1]+1, dp[i-1][j]+1);
                }

            }
        }
        
        return dp[m][n];
    }

    

    int Min(int a, int b, int c){
        return min(a, min(b,c));
    }
};
```

### 931 给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。

下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。在下一行选择的元素和当前行所选元素最多相隔一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/minimum-falling-path-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


>  备忘录
```
class Solution {
public:
    vector<vector<int>> memo;
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int n = matrix.size();
        vector<vector<int>> mat(n, vector<int>(n, 66666));
        memo = mat;
        int res = INT_MAX;
        for(int i=0;i<n;i++){
            res = min(res, dp(matrix, n-1,i));
        }
        return res;
    }

    int dp(vector<vector<int>> matrix, int i, int j){
        if (i<0 || j<0 || i>=matrix.size()|| j>=matrix[0].size()) return 99999;
        if (i==0) return matrix[i][j];
        if (memo[i][j]!=66666) return  memo[i][j];

        memo[i][j] = matrix[i][j] + Min(dp(matrix,i-1,j), dp(matrix,i-1,j-1), dp(matrix,i-1,j+1) );

        return memo[i][j];
    }


    int Min(int a, int b, int c){
        return min(a, min(b,c));
    }
};
```

> DP
https://leetcode.cn/problems/minimum-falling-path-sum/solution/xia-jiang-lu-jing-zui-xiao-he-by-leetcode/


算法

我们可以直接在原数组 A 上计算 dp(r, c)，因为最后一行 A 的值就是 dp 的值 。因此我们从倒数第二行开始，从下往上进行动态规划，状态转移方程为：

A[r][c] = A[r][c] + min{A[r + 1][c - 1], A[r + 1][c], A[r + 1][c + 1]}

注意需要处理超出边界的情况，即在第一列和最后一列只能下降到两个位置。

作者：LeetCode
链接：https://leetcode.cn/problems/minimum-falling-path-sum/solution/xia-jiang-lu-jing-zui-xiao-he-by-leetcode/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```
class Solution {
public:
    int minFallingPathSum(vector<vector<int>>& matrix) {
        int N = matrix.size();
        
        for (int r = N-2; r >= 0; --r) {
            for (int c = 0; c < N; ++c) {
                // best = min(A[r+1][c-1], A[r+1][c], A[r+1][c+1])
                int best = matrix[r+1][c];
                if (c > 0)
                    best = min(best, matrix[r+1][c-1]);
                if (c+1 < N)
                    best = min(best, matrix[r+1][c+1]);
                matrix[r][c] += best;
            }
        }


        int res = INT_MAX;
        for(int i=0;i<N;i++){
            res = min(res, matrix[0][i]);
        }
        return res;
    }

};
```
### 55. 跳跃游戏

给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

 

示例 1：

输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/jump-game
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


```
    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int k=0;

        for (int i = 0; i < n; ++i) {
            if (i <= k) {
                k = max(k, i + nums[i]);
                if (k >= n - 1) {
                    return true;
                }
            }
        }
        return false;

    }
```

这种方法所依据的核心特性：如果一个位置能够到达，那么这个位置左侧所有位置都能到达。 想到这一点，解法就呼之欲出了~

```
    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int k=0;
        for(int i=0;i<n;i++){
            if (k<i) return false;
            k = max( k, i + nums[i]);
        }
        return true;
    }
```

### 45. 跳跃游戏 II

给你一个非负整数数组 nums ，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。

 

示例 1:

输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/jump-game-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


懂和不懂我觉得就是差这几行注释： 思想就一句话：每次在上次能跳到的范围（end）内选择一个能跳的最远的位置（也就是能跳到max_far位置的点）作为下次的起跳点 ！
```
class Solution {
public:
    int jump(vector<int>& nums) 
    {
        int max_far = 0;// 目前能跳到的最远位置
        int step = 0;   // 跳跃次数
        int end = 0;    // 上次跳跃可达范围右边界（下次的最右起跳点）
        for (int i = 0; i < nums.size() - 1; i++)
        {
            max_far = std::max(max_far, i + nums[i]);
            // 到达上次跳跃能到达的右边界了
            if (i == end)
            {
                end = max_far;  // 目前能跳到的最远位置变成了下次起跳位置的有边界
                step++;         // 进入下一次跳跃
            }
        }
        return step;
    }
};
```


### 70. 爬楼梯

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

示例 1：

输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶

```
class Solution {
public:
    int climbStairs(int n) {
        vector<int> dp(n+1);
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<=n;i++){
            dp[i]=dp[i-1]+dp[i-2];
        }
        return dp[n];
    }
};
```

### 746. 使用最小花费爬楼梯
给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。

你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。

请你计算并返回达到楼梯顶部的最低花费。

```
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        // 创建长度为 n+1 的数组dp，其中 dp[i] 表示达到下标 i 的最小花费。
        vector<int> dp(n+1);
        dp[0] = dp[1] = 0;
        for(int i=2;i<=n;i++){
            dp[i] = min(dp[i-1] + cost[i-1], dp[i-2]+cost[i-2] );
        }
        return dp[n];
    }
};
```


## 198. 打家劫舍
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

 

示例 1：

输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/house-robber
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


首先考虑最简单的情况。如果只有一间房屋，则偷窃该房屋，可以偷窃到最高总金额。如果只有两间房屋，则由于两间房屋相邻，不能同时偷窃，只能偷窃其中的一间房屋，因此选择其中金额较高的房屋进行偷窃，可以偷窃到最高总金额。

如果房屋数量大于两间，应该如何计算能够偷窃到的最高总金额呢？对于第 k~(k>2)k (k>2) 间房屋，有两个选项：

偷窃第 k 间房屋，那么就不能偷窃第 k−1 间房屋，偷窃总金额为前 k−2 间房屋的最高总金额与第 k 间房屋的金额之和。

不偷窃第 k 间房屋，偷窃总金额为前 k−1 间房屋的最高总金额。

在两个选项中选择偷窃总金额较大的选项，该选项对应的偷窃总金额即为前 k 间房屋能偷窃到的最高总金额。

用 dp[i] 表示前 i 间房屋能偷窃到的最高总金额，那么就有如下的状态转移方程：
dp[i]=max(dp[i−2]+nums[i],dp[i−1])

边界条件为：
- 只有一间房屋，则偷窃该房屋 
- 只有两间房屋，选择其中金额较高的房屋进行偷窃 


最终的答案即为 dp[n−1]，其中 nn 是数组的长度。

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/house-robber/solution/da-jia-jie-she-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
class Solution {
public:
    int rob(vector<int>& nums) {
        if (nums.empty()) {
            return 0;
        }
        int n = nums.size();
        if (n == 1) {
            return nums[0];
        }
        vector<int> dp(n);
        dp[0] = nums[0];
        dp[1] = max(nums[0], nums[1]);
        for(int i=2;i<n;i++){
            dp[i] = max(dp[i-2]+nums[i], dp[i-1]);
        }
        return dp[n-1];
    }
};
```


### 213. 打家劫舍 II

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

 

示例 1：

输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/house-robber-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```
class Solution {
public:

    int dp(vector<int> nums, int start, int end){
        int first = nums[start];
        int second = max(nums[start], nums[start+1]);
        for(int i=start+2;i<=end;i++){
            int tmp = second;
            second = max(first+nums[i], second);
            first = tmp;
        }
        return second;
    }

    int rob(vector<int>& nums) {
        if (nums.empty()) {
            return 0;
        }
        int n = nums.size();
        if (n == 1) {
            return nums[0];
        }
        if (n==2){
            return max(nums[0],nums[1]);
        }
        return max(dp(nums, 0, n-2), dp(nums, 1, n-1 ));
    }
};
```

### 337. 打家劫舍 III

小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。

除了 root 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。

给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/house-robber-iii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

输入: root = [3,2,3,null,3,null,1]
输出: 7 
解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7

> 解题思路

```
简化一下这个问题：一棵二叉树，树上的每个点都有对应的权值，每个点有两种状态（选中和不选中），问在不能同时选中有父子关系的点的情况下，能选中的点的最大权值和是多少。

我们可以用 f(o) 表示选择 o 节点的情况下，o 节点的子树上被选择的节点的最大权值和；g(o) 表示不选择 o 节点的情况下，o 节点的子树上被选择的节点的最大权值和；l 和 r 代表 oo 的左右孩子。

- 当 o 被选中时，o 的左右孩子都不能被选中，故 o 被选中情况下子树上被选中点的最大权值和为 l 和 r 不被选中的最大权值和相加，即 f(o) = g(l) + g(r)。
- 当 o 不被选中时，o 的左右孩子可以被选中，也可以不被选中。对于 o 的某个具体的孩子 x，它对 o 的贡献是 x 被选中和不被选中情况下权值和的较大值。故 g(o)=max{f(l),g(l)}+max{f(r),g(r)}。


至此，我们可以用哈希表来存 f 和 g 的函数值，用深度优先搜索的办法后序遍历这棵二叉树，我们就可以得到每一个节点的 f 和 g。根节点的 ff 和 gg 的最大值就是我们要找的答案。

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/house-robber-iii/solution/da-jia-jie-she-iii-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
class Solution {
public:
    unordered_map<TreeNode* , int > f,g;

    void dfs(TreeNode* root){
        if(!root) return;

        dfs(root->left);
        dfs(root->right);
        f[root] = root->val + g[root->left] + g[root->right];
        g[root] = max(f[root->left], g[root->left]) + max(f[root->right], g[root->right]);
    }


    int rob(TreeNode* root) {
        dfs(root);
        return max(f[root], g[root]);
    }
};
```
