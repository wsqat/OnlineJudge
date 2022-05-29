## 背包



我觉得 Two Sum 系列问题就是想教我们如何使用哈希表处理问题。我们接着往后看。



## 1. 两数之和

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

 

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/two-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


### 借助hashmap
```
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        unordered_map<int, int> index;
        for(int i=0;i<n;i++){
            index[nums[i]]=i;
        }
        for(int i=0;i<n;i++){
            if(index.count(target-nums[i]) && i!=index[target-nums[i]] ){
                return {i, index[target-nums[i]]};
            }
        }
        return {-1,-1};
    }
};
```

## 15. 三数之和

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

 

示例 1：

输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/3sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


```
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) 
    {
        int size = nums.size();
        if (size < 3)   return {};          // 特判
        vector<vector<int> >res;            // 保存结果（所有不重复的三元组）
        std::sort(nums.begin(), nums.end());// 排序（默认递增）
        for (int i = 0; i < size; i++)      // 固定第一个数，转化为求两数之和
        {
            if (nums[i] > 0)    return res; // 第一个数大于 0，后面都是递增正数，不可能相加为零了
            // 去重：如果此数已经选取过，跳过
            if (i > 0 && nums[i] == nums[i-1])  continue;
            // 双指针在nums[i]后面的区间中寻找和为0-nums[i]的另外两个数
            int left = i + 1;
            int right = size - 1;
            while (left < right)
            {
                if (nums[left] + nums[right] > -nums[i])
                    right--;    // 两数之和太大，右指针左移
                else if (nums[left] + nums[right] < -nums[i])
                    left++;     // 两数之和太小，左指针右移
                else
                {
                    // 找到一个和为零的三元组，添加到结果中，左右指针内缩，继续寻找
                    res.push_back(vector<int>{nums[i], nums[left], nums[right]});
                    left++;
                    right--;
                    // 去重：第二个数和第三个数也不重复选取
                    // 例如：[-4,1,1,1,2,3,3,3], i=0, left=1, right=5
                    while (left < right && nums[left] == nums[left-1])  left++;
                    while (left < right && nums[right] == nums[right+1])    right--;
                }
            }
        }
        return res;
    }
};
```


## 18. 四数之和

给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]] （若两个四元组元素一一对应，则认为两个四元组重复）：

0 <= a, b, c, d < n
a、b、c 和 d 互不相同
nums[a] + nums[b] + nums[c] + nums[d] == target
你可以按 任意顺序 返回答案 。

 

示例 1：

输入：nums = [1,0,-1,0,-2,2], target = 0
输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/4sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

> 思路
```
最朴素的方法是使用四重循环枚举所有的四元组，然后使用哈希表进行去重操作，得到不包含重复四元组的最终答案。假设数组的长度是 nn，则该方法中，枚举的时间复杂度为 O(n 
4
 )，去重操作的时间复杂度和空间复杂度也很高，因此需要换一种思路。

为了避免枚举到重复四元组，则需要保证每一重循环枚举到的元素不小于其上一重循环枚举到的元素，且在同一重循环中不能多次枚举到相同的元素。

为了实现上述要求，可以对数组进行排序，并且在循环过程中遵循以下两点：

每一种循环枚举到的下标必须大于上一重循环枚举到的下标；

同一重循环中，如果当前元素与上一个元素相同，则跳过当前元素。

使用上述方法，可以避免枚举到重复四元组，但是由于仍使用四重循环，时间复杂度仍是 O(n 
4
 )。注意到数组已经被排序，因此可以使用双指针的方法去掉一重循环。

使用两重循环分别枚举前两个数，然后在两重循环枚举到的数之后使用双指针枚举剩下的两个数。假设两重循环枚举到的前两个数分别位于下标 i 和 j，其中 i<j。初始时，左右指针分别指向下标 j+1 和下标 n−1。每次计算四个数的和，并进行如下操作：

如果和等于 target，则将枚举到的四个数加到答案中，然后将左指针右移直到遇到不同的数，将右指针左移直到遇到不同的数；

如果和小于 target，则将左指针右移一位；

如果和大于 target，则将右指针左移一位。

作者：LeetCode-Solution
链接：https://leetcode.cn/problems/4sum/solution/si-shu-zhi-he-by-leetcode-solution/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


```
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
       vector<vector<int> > res;
       int n = nums.size();
       if (n < 4)   return res; // 特判
       std::sort(nums.begin(), nums.end()); // 排序
       // 选取第一个数
       for (int i = 0; i < n - 3; i++)
       {
            if (i > 0 && nums[i] == nums[i-1])  // 去重
                continue;
            if ((long) nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target) // 剪枝 + int类型转long
                break;
            if ((long) nums[i] + nums[n-3] + nums[n-2] + nums[n-1] < target) // 剪枝 + int类型转long
                continue;
            // 选取第二个数
            for (int j = i+1; j < n - 2; j++)
            {
                if (j > i+1 && nums[j] == nums[j-1]) // 去重
                    continue;
                if ((long) nums[i] + nums[j] + nums[j+1] + nums[j+2] > target) // 剪枝  + int类型转long
                    break;
                if ((long) nums[i] + nums[j] + nums[n-2] + nums[n-1] < target) // 剪枝 + int类型转long
                    continue;               
                // 通过双指针取第三个和第四个数
                int left = j + 1;
                int right = n - 1;
                while (left < right)
                {
                    if (nums[left] + nums[right] < target - nums[i] - nums[j])
                        left++;     // 此两数之和太小，左指针右移
                    else if (nums[left] + nums[right] > target - nums[i] - nums[j])
                        right--;    // 此两数之和太大，右指针左移
                    else
                    {
                        // 找到一组解，左右指针内缩，继续寻找
                        res.push_back(vector<int>{nums[i], nums[j], nums[left], nums[right]});
                        left++; right--;
                        // 去重
                        while (left < right && nums[left] == nums[left-1])   left++;
                        while (left < right && nums[right] == nums[right+1])   right--;
                    }
                }
            }
       } 
       return res;
    }
}; 
```
