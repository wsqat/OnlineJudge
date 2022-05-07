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
