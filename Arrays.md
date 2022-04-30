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
