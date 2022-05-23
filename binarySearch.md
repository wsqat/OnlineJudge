# 我写了⾸诗，把⼆分搜索变成了默写题

## 零、⼆分查找框架
```
int binarySearch(int[] nums, int target) {
 int left = 0, right = ...;
 while(...) {
 int mid = left + (right - left) / 2;
 if (nums[mid] == target) {
 ...
 } else if (nums[mid] < target) {
 left = ...
 } else if (nums[mid] > target) {
 right = ...
 }
 }
 return ...;
}
```

分析⼆分查找的⼀个技巧是：不要出现 else，⽽是把所有情况⽤ else if 写清楚，这样可以清楚地展现所有细
节。本⽂都会使⽤ else if，旨在讲清楚，读者理解后可⾃⾏简化。

另外提前说明⼀下，计算 mid 时需要防⽌溢出，代码中 left + (right - left) / 2 就和 (left +
right) / 2 的结果相同，但是有效防⽌了 left 和 right 太⼤，直接相加导致溢出的情况。

## ⼀、寻找⼀个数（基本的⼆分搜索）

```
int binarySearch(int[] nums, int target) {
 int left = 0; 
 int right = nums.length - 1; // 注意
 while(left <= right) {
 int mid = left + (right - left) / 2;
 if(nums[mid] == target)
 return mid; 
 else if (nums[mid] < target)
 left = mid + 1; // 注意
 else if (nums[mid] > target)
 right = mid - 1; // 注意
 }
 return -1;
}

```

## ⼆、寻找左侧边界的⼆分搜索

```
int left_bound(int[] nums, int target) {
 int left = 0, right = nums.length - 1;
 // 搜索区间为 [left, right]
 while (left <= right) {
 int mid = left + (right - left) / 2;
 if (nums[mid] < target) {
 // 搜索区间变为 [mid+1, right]
 left = mid + 1;
 } else if (nums[mid] > target) {
 // 搜索区间变为 [left, mid-1]
 right = mid - 1;
 } else if (nums[mid] == target) {
 // 收缩右侧边界
 right = mid - 1;
 }
 }
 // 检查出界情况
 if (left >= nums.length || nums[left] != target) {
 return -1;
 }
 return left;
}

```

## 三、寻找右侧边界的⼆分查找

类似搜索左侧边界的统⼀写法，其实只要改两个地⽅就⾏了：
```
int right_bound(int[] nums, int target) {
 int left = 0, right = nums.length - 1;
 while (left <= right) {
 int mid = left + (right - left) / 2;
 if (nums[mid] < target) {
 left = mid + 1;
 } else if (nums[mid] > target) {
 right = mid - 1;
 } else if (nums[mid] == target) {
 // 这⾥改成收缩左侧边界即可
 left = mid + 1;
 }
 }
 // 这⾥改为检查 right 越界的情况，⻅下图
 if (right < 0 || nums[right] != target) {
 return -1;
 }
 return right;
}
```
当 target ⽐所有元素都⼩时，right 会被减到 -1，所以需要在最后防⽌越界：


```
int binary_search(int[] nums, int target) {
 int left = 0, right = nums.length - 1; 
 while(left <= right) {
 int mid = left + (right - left) / 2;
 if (nums[mid] < target) {
 left = mid + 1;
 } else if (nums[mid] > target) {
 right = mid - 1; 
 } else if(nums[mid] == target) {
 // 直接返回
 return mid;
 }
 }
 // 直接返回
 return -1;
}
int left_bound(int[] nums, int target) {
 int left = 0, right = nums.length - 1;
 while (left <= right) {
 int mid = left + (right - left) / 2;
 if (nums[mid] < target) {
 left = mid + 1;
 } else if (nums[mid] > target) {
 right = mid - 1;
 } else if (nums[mid] == target) {
 // 别返回，锁定左侧边界
 right = mid - 1;
 }
 }
 // 最后要检查 left 越界的情况
 if (left >= nums.length || nums[left] != target) {
 return -1;
在线⽹站 labuladong的刷题三件套
97 / 682
 }
 return left;
}
int right_bound(int[] nums, int target) {
 int left = 0, right = nums.length - 1;
 while (left <= right) {
 int mid = left + (right - left) / 2;
 if (nums[mid] < target) {
 left = mid + 1;
 } else if (nums[mid] > target) {
 right = mid - 1;
 } else if (nums[mid] == target) {
 // 别返回，锁定右侧边界
 left = mid + 1;
 }
 }
 // 最后要检查 right 越界的情况
 if (right < 0 || nums[right] != target) {
 return -1;
 }
 return right;
}
```


果以上内容你都能理解，那么恭喜你，⼆分查找算法的细节不过如此。
通过本⽂，你学会了：
1、分析⼆分查找代码时，不要出现 else，全部展开成 else if ⽅便理解。
2、注意「搜索区间」和 while 的终⽌条件，如果存在漏掉的元素，记得在最后检查。
3、如需定义左闭右开的「搜索区间」搜索左右边界，只要在 nums[mid] == target 时做修改即可，搜索
右侧时需要减⼀。
4、如果将「搜索区间」全都统⼀成两端都闭，好记，只要稍改 nums[mid] == target 条件处的代码和返
回的逻辑即可，推荐拿⼩本本记下，作为⼆分搜索模板。

## 704. 二分查找
```
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left=0,right=nums.size()-1;
        while(left<=right){
            int mid = (right+left)/2;
            if(nums[mid]==target){
                return mid;
            }else if(nums[mid]<target){
                left = mid+1;
            }else{
                right = mid-1;
            }
        }
        return -1;
    }
};
```

## 34. 在排序数组中查找元素的第一个和最后一个位置

```
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if(nums.empty()) return {-1,-1};
    
        int l = 0, r = nums.size() - 1; //二分范围
        while( l < r)			        //查找元素的开始位置
        {
            int mid = (l + r )/2;
            if(nums[mid] >= target) r = mid;
            else l = mid + 1;
        }
        if( nums[r] != target) return {-1,-1};  //查找失败
        int L = r;
        l = 0, r = nums.size() - 1;     //二分范围
        while( l < r)                   //查找元素的结束位置
        {
            int mid = (l + r + 1)/2;
            if(nums[mid] <= target ) l = mid;
            else r = mid - 1;
        }
        return {L,r};
    }
};

作者：lin-shen-shi-jian-lu-k
链接：https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/solution/tu-jie-er-fen-zui-qing-xi-yi-dong-de-jia-ddvc/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

```
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if (nums.empty()) return {-1,-1};
        int l=0,r=nums.size()-1;
        while(l<r){
            int mid = (l+r)/2;
            if (nums[mid]>=target) r=mid;
            else l=mid+1;
        }
        if (nums[r]!=target) return {-1,-1};

        int L = r;
        l = 0, r = nums.size()-1;
        while(l<r){
            int mid = (l+r+1)/2;
            if (nums[mid]<=target) l=mid;
            else r=mid-1;
        }
        return {L,r};
    }
};
```
