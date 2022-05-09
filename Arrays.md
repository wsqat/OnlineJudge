https://codetop.cc/discuss/algo

# 一. 前缀和专题
> 前缀和主要适⽤的场景是原始数组不会被修改的情况下，频繁查询某个区间的累加和。

## 303. 区域和检索 - 数组不可变
给定一个整数数组  nums，处理以下类型的多个查询:

计算索引 left 和 right （包含 left 和 right）之间的 nums 元素的 和 ，其中 left <= right
实现 NumArray 类：

NumArray(int[] nums) 使用数组 nums 初始化对象
int sumRange(int i, int j) 返回数组 nums 中索引 left 和 right 之间的元素的 总和 ，包含 left 和 right 两点（也就是 nums[left] + nums[left + 1] + ... + nums[right] )

### example
```
输入：
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
输出：
[null, 1, -1, -3]

解释：
NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
numArray.sumRange(0, 2); // return 1 ((-2) + 0 + 3)
numArray.sumRange(2, 5); // return -1 (3 + (-5) + 2 + (-1)) 
numArray.sumRange(0, 5); // return -3 ((-2) + 0 + 3 + (-5) + 2 + (-1))
```
### resolution
```
前缀和
最朴素的想法是存储数组 nums 的值，每次调用sumRange 时，通过循环的方法计算数组 nums 从下标 i 到下标 j 范围内的元素和，需要计算 j-i+1 个元素的和。由于每次检索的时间和检索的下标范围有关，因此检索的时间复杂度较高，如果检索次数较多，则会超出时间限制。

由于会进行多次检索，即多次调用 sumRange，因此为了降低检索的总时间，应该降低sumRange 的时间复杂度，最理想的情况是时间复杂度 O(1)。为了将检索的时间复杂度降到 O(1)，需要在初始化的时候进行预处理。
```

### code
link :  https://leetcode-cn.com/problems/range-sum-query-immutable/solution/qu-yu-he-jian-suo-shu-zu-bu-ke-bian-by-l-px41/
```
class NumArray {
public:
    vector<int> preSum;

public:
    NumArray(vector<int>& nums) {
        int n = nums.size();
        preSum.resize(n+1);
        for(int i=0; i< n; i++){
            preSum[i+1] = preSum[i]+nums[i];
        }
    }
    
    int sumRange(int left, int right) {
        return preSum[right+1]-preSum[left];
    }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * int param_1 = obj->sumRange(left,right);
 */
  ```



## 304. 二维区域和检索 - 矩阵不可变
给定一个二维矩阵 matrix，以下类型的多个请求：

计算其子矩形范围内元素的总和，该子矩阵的 左上角 为 (row1, col1) ，右下角 为 (row2, col2) 。
实现 NumMatrix 类：

NumMatrix(int[][] matrix) 给定整数矩阵 matrix 进行初始化
int sumRegion(int row1, int col1, int row2, int col2) 返回 左上角 (row1, col1) 、右下角 (row2, col2) 所描述的子矩阵的元素 总和 。

### example
```
输入: 
["NumMatrix","sumRegion","sumRegion","sumRegion"]
[[[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]],[2,1,4,3],[1,1,2,2],[1,2,2,4]]
输出: 
[null, 8, 11, 12]

解释:
NumMatrix numMatrix = new NumMatrix([[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]);
numMatrix.sumRegion(2, 1, 4, 3); // return 8 (红色矩形框的元素总和)
numMatrix.sumRegion(1, 1, 2, 2); // return 11 (绿色矩形框的元素总和)
numMatrix.sumRegion(1, 2, 2, 4); // return 12 (蓝色矩形框的元素总和)

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/range-sum-query-2d-immutable
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```
### resolution
```
定义积分图integeral[i][j]，表示前i行,j列的元素组成的和
那么
integeral[i][j]=integeral[i-1][j]+integeral[i][j-1]-integeral[i-1][j-1]+matrix[i-1][j-1];
某个区间的和可以用，积分图的差进行计算，主要要加上多减的部分
integeral[row2+1][col2+1]-integeral[row1][col2+1]-integeral[row2+1][col1]+integeral[row1][col1];

```

### code
link :  https://leetcode-cn.com/problems/range-sum-query-2d-immutable/
```
class NumMatrix {

public:
    vector<vector<int>> preSum;

    NumMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        
        preSum=vector<vector<int>>(m+1,vector<int>(n+1,0));
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                preSum[i][j] = preSum[i-1][j] + preSum[i][j-1] + matrix[i-1][j-1] - preSum[i-1][j-1]; 
            }
        }
    }
    
    int sumRegion(int row1, int col1, int row2, int col2) {
        return preSum[row2+1][col2+1] - preSum[row1][col2+1] - preSum[row2+1][col1] + preSum[row1][col1];
    }
};

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix* obj = new NumMatrix(matrix);
 * int param_1 = obj->sumRegion(row1,col1,row2,col2);
 */
  ```



## 560. 和为 K 的子数组

给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。


### example
```
示例 1：

输入：nums = [1,1,1], k = 2
输出：2
示例 2：

输入：nums = [1,2,3], k = 3
输出：2


提示：

1 <= nums.length <= 2 * 104
-1000 <= nums[i] <= 1000
-107 <= k <= 107

```

### code
link :  https://leetcode-cn.com/problems/subarray-sum-equals-k/submissions/
```
class Solution {
public:

// 定义前缀和，sumval[i]表示，前i个数字的和。
// 又由于sumval-（sumval-k）=k，因此只需要统计多少个sumval-k即可。
// 利用hash统计前缀和的个数。
// 注意：
// 当选择0个数字时，sumval[0]=0，hash[0]=1

    
    int subarraySum(vector<int>& nums, int k) {
        int n = nums.size();
        //前缀和
        unordered_map<int,int> preSum;
        preSum[0]=1;
        int sumVal = 0;
        int cnt = 0;

        for(int v: nums){
            sumVal += v;
            
            if(preSum.count(sumVal-k)){
                cnt+=preSum[sumVal-k];
            }
            preSum[sumVal]+=1;
        }


        return cnt;
    }
};
```

# 二. 差分和
prefix[i] 就代表着 nums[0..i-1] 所有元素的累加和，如果我们想求区间 nums[i..j] 的累加和，只要
计算 prefix[j+1] - prefix[i] 即可，⽽不需要遍历整个区间求和。
本⽂讲⼀个和前缀和思想⾮常类似的算法技巧「差分数组」，差分数组的主要适⽤场景是频繁对原始数组的
某个区间的元素进⾏增减。


这⾥就需要差分数组的技巧，类似前缀和技巧构造的 prefix 数组，我们先对 nums 数组构造⼀个 diff 差
分数组，diff[i] 就是 nums[i] 和 nums[i-1] 之差：

```
通过这个 diff 差分数组是可以反推出原始数组 nums 的，代码逻辑如下：
int[] res = new int[diff.length];
// 根据差分数组构造结果数组
res[0] = diff[0];
for (int i = 1; i < diff.length; i++) {
 res[i] = res[i - 1] + diff[i];
}
这样构造差分数组 diff，就可以快速进⾏区间增减的操作，如果你想对区间 nums[i..j] 的元素全部加
3，那么只需要让 diff[i] += 3，然后再让 diff[j+1] -= 3 即可：
```

原理很简单，回想 diff 数组反推 nums 数组的过程，diff[i] += 3 意味着给 nums[i..] 所有的元素都
加了 3，然后 diff[j+1] -= 3 ⼜意味着对于 nums[j+1..] 所有元素再减 3，那综合起来，是不是就是对
nums[i..j] 中的所有元素都加 3 了？
只要花费 O(1) 的时间修改 diff 数组，就相当于给 nums 的整个区间做了修改。多次修改 diff，然后通过
diff 数组反推，即可得到 nums 修改后的结果。

```
差分数组的性质这段话想了半天,终于理解了,为了仅仅让原数组下标在[l,r]区间内的元素值增加 inc

首先需要将差分数组d[l]位置加 inc,这样下标 >= l 位置的元素都获得inc增量

其次,为了不影响原数组中下标大于 r 的元素,需要在d[r + 1]处减去inc,使得原数组下标 > r 的元素值不变.
```

## 1109. 航班预订统计

这里有 n 个航班，它们分别从 1 到 n 进行编号。

有一份航班预订表 bookings ，表中第 i 条预订记录 bookings[i] = [firsti, lasti, seatsi] 意味着在从 firsti 到 lasti （包含 firsti 和 lasti ）的 每个航班 上预订了 seatsi 个座位。

请你返回一个长度为 n 的数组 answer，里面的元素是每个航班预定的座位总数。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/corporate-flight-bookings
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

### example
```
示例 1：

输入：bookings = [[1,2,10],[2,3,20],[2,5,25]], n = 5
输出：[10,55,45,25,25]
解释：
航班编号        1   2   3   4   5
预订记录 1 ：   10  10
预订记录 2 ：       20  20
预订记录 3 ：       25  25  25  25
总座位数：      10  55  45  25  25
因此，answer = [10,55,45,25,25]
示例 2：

输入：bookings = [[1,2,10],[2,2,15]], n = 2
输出：[10,25]
解释：
航班编号        1   2
预订记录 1 ：   10  10
预订记录 2 ：       15
总座位数：      10  25
因此，answer = [10,25]
 

提示：

1 <= n <= 2 * 104
1 <= bookings.length <= 2 * 104
bookings[i].length == 3
1 <= firsti <= lasti <= n
1 <= seatsi <= 104

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/corporate-flight-bookings
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

### code
```
class Solution {
public:
    vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
        vector<int> diff(n);
        int m = bookings.size();
        for(vector<int> book: bookings){
            diff[book[0]-1] += book[2];
            if (book[1]<n){
                diff[book[1]] -= book[2];
            }
        }

        for(int i=1;i<n;i++){
            diff[i] += diff[i-1];
        }
        return diff;
    }
};


复杂度分析

时间复杂度：O(n+m)O(n+m)，其中 nn 为要求的数组长度，mm 为预定记录的数量。我们需要对于每一条预定记录处理一次差分数组，并最后对差分数组求前缀和。

空间复杂度：O(1)O(1)。我们只需要常数的空间保存若干变量，注意返回值不计入空间复杂度

作者：LeetCode-Solution
链接：https://leetcode-cn.com/problems/corporate-flight-bookings/solution/hang-ban-yu-ding-tong-ji-by-leetcode-sol-5pv8/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


## 1094. 拼车

车上最初有 capacity 个空座位。车 只能 向一个方向行驶（也就是说，不允许掉头或改变方向）

给定整数 capacity 和一个数组 trips ,  trip[i] = [numPassengersi, fromi, toi] 表示第 i 次旅行有 numPassengersi 乘客，接他们和放他们的位置分别是 fromi 和 toi 。这些位置是从汽车的初始位置向东的公里数。

当且仅当你可以在所有给定的行程中接送所有乘客时，返回 true，否则请返回 false。

 
```
示例 1：

输入：trips = [[2,1,5],[3,3,7]], capacity = 4
输出：false
示例 2：

输入：trips = [[2,1,5],[3,3,7]], capacity = 5
输出：true
 

提示：

1 <= trips.length <= 1000
trips[i].length == 3
1 <= numPassengersi <= 100
0 <= fromi < toi <= 1000
1 <= capacity <= 105

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/car-pooling
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

### 解题思路
```
差分数组

按照trips信息去构建一个数组，对应每个点乘客的变化
从起点开始按照差分去计算当前乘客，超过capacity则直接返回
```
### code
```
class Solution {
public:
    bool carPooling(vector<vector<int>>& trips, int capacity) {
        int m = 1001;
        vector<int> diff(m);
        for(auto& trip: trips){
            diff[trip[1]] += trip[0];
            diff[trip[2]] -= trip[0];
        }
        if(diff[0]>capacity)
            return false;
        
        for(int i=1; i<m;i++){
            diff[i] += diff[i-1];
            if(diff[i]>capacity)
                return false;
        }
        return true;
    }
};
```
