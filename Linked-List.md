## 21. 合并两个有序链表

https://leetcode-cn.com/problems/merge-two-sorted-lists/solution/he-bing-liang-ge-you-xu-lian-biao-by-leetcode-solu/

将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

### 方法一：递归

我们可以如下递归地定义两个链表里的 merge 操作（忽略边界情况，比如空链表等）：
 
list1[0]+merge(list1[1:],list2);  list1[0] < list2[0]
list2[0]+merge(list1,list2[1:]);  otherwise

也就是说，两个链表头部值较小的一个节点与剩下元素的 merge 操作结果合并。

算法

我们直接将以上递归过程建模，同时需要考虑边界情况。

如果 l1 或者 l2 一开始就是空链表 ，那么没有任何操作需要合并，所以我们只需要返回非空链表。否则，我们要判断 l1 和 l2 哪一个链表的头节点的值更小，然后递归地决定下一个添加到结果里的节点。如果两个链表有一个为空，递归结束。


```
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (l1 == nullptr) {
            return l2;
        } else if (l2 == nullptr) {
            return l1;
        } else if (l1->val < l2->val) {
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        } else {
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
};
```
复杂度分析

时间复杂度：O(n + m)，其中 n 和 m 分别为两个链表的长度。因为每次调用递归都会去掉 l1 或者 l2 的头节点（直到至少有一个链表为空），函数 mergeTwoList 至多只会递归调用每个节点一次。因此，时间复杂度取决于合并后的链表长度，即 O(n+m)。

空间复杂度：O(n + m)O，其中 n 和 m 分别为两个链表的长度。递归调用 mergeTwoLists 函数时需要消耗栈空间，栈空间的大小取决于递归调用的深度。结束递归调用时 mergeTwoLists 函数最多调用 n+mn+m 次，因此空间复杂度为 O(n+m)。

### 方法二：迭代

我们可以用迭代的方法来实现上述算法。当 l1 和 l2 都不是空链表时，判断 l1 和 l2 哪一个链表的头节点的值更小，将较小值的节点添加到结果里，当一个节点被添加到结果里之后，将对应链表中的节点向后移一位。

算法

首先，我们设定一个哨兵节点 prehead ，这可以在最后让我们比较容易地返回合并后的链表。我们维护一个 prev 指针，我们需要做的是调整它的 next 指针。然后，我们重复以下过程，直到 l1 或者 l2 指向了 null ：如果 l1 当前节点的值小于等于 l2 ，我们就把 l1 当前的节点接在 prev 节点的后面同时将 l1 指针往后移一位。否则，我们对 l2 做同样的操作。不管我们将哪一个元素接在了后面，我们都需要把 prev 向后移一位。

在循环终止的时候， l1 和 l2 至多有一个是非空的。由于输入的两个链表都是有序的，所以不管哪个链表是非空的，它包含的所有元素都比前面已经合并链表中的所有元素都要大。这意味着我们只需要简单地将非空链表接在合并链表的后面，并返回合并链表即可。

下图展示了 1->4->5 和 1->2->3->6 两个链表迭代合并的过程：

```

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* preHead = new ListNode(-1);

        ListNode* prev = preHead;
        while (l1 != nullptr && l2 != nullptr) {
            if (l1->val < l2->val) {
                prev->next = l1;
                l1 = l1->next;
            } else {
                prev->next = l2;
                l2 = l2->next;
            }
            prev = prev->next;
        }

        // 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
        prev->next = l1 == nullptr ? l2 : l1;

        return preHead->next;
    }
};
复杂度分析

时间复杂度：O(n + m)O(n+m)，其中 nn 和 mm 分别为两个链表的长度。因为每次循环迭代中，l1 和 l2 只有一个元素会被放进合并链表中， 因此 while 循环的次数不会超过两个链表的长度之和。所有其他操作的时间复杂度都是常数级别的，因此总的时间复杂度为 O(n+m)O(n+m)。

空间复杂度：O(1)O(1)。我们只需要常数的空间存放若干变量。
```
##  23. 合并K个升序链表

https://leetcode-cn.com/problems/merge-k-sorted-lists/solution/he-bing-kge-pai-xu-lian-biao-by-leetcode-solutio-2/

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。


### 方法一：顺序合并

我们可以想到一种最朴素的方法：用一个变量 ans 来维护以及合并的链表，第 i 次循环把第 i 个链表和 ans 合并，答案保存到 ans 中。

```
class Solution {
public:
    ListNode* mergeTwoLists(ListNode *a, ListNode *b) {
        if ((!a) || (!b)) return a ? a : b;
        ListNode head, *tail = &head, *aPtr = a, *bPtr = b;
        while (aPtr && bPtr) {
            if (aPtr->val < bPtr->val) {
                tail->next = aPtr; aPtr = aPtr->next;
            } else {
                tail->next = bPtr; bPtr = bPtr->next;
            }
            tail = tail->next;
        }
        tail->next = (aPtr ? aPtr : bPtr);
        return head.next;
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode *ans = nullptr;
        for (size_t i = 0; i < lists.size(); ++i) {
            ans = mergeTwoLists(ans, lists[i]);
        }
        return ans;
    }
};
```

### 方法二：分治合并
考虑优化方法一，用分治的方法进行合并。


```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:

    ListNode* mergeTwoLists(ListNode *a, ListNode *b) {
        if ((!a) || (!b)) return a ? a : b;
        ListNode head, *tail = &head, *aPtr = a, *bPtr = b;
        while (aPtr && bPtr) {
            if (aPtr->val < bPtr->val) {
                tail->next = aPtr; aPtr = aPtr->next;
            } else {
                tail->next = bPtr; bPtr = bPtr->next;
            }
            tail = tail->next;
        }
        tail->next = (aPtr ? aPtr : bPtr);
        return head.next;
    }

    ListNode* merge(vector <ListNode*> &lists, int l, int r) {
        if (l == r) return lists[l];
        if (l > r) return nullptr;
        int mid = (l + r) >> 1;
        return mergeTwoLists(merge(lists, l, mid), merge(lists, mid + 1, r));
    }

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return merge(lists, 0, lists.size() - 1);
    }
};
```

### 方法三：使用优先队列合并

这个方法和前两种方法的思路有所不同，我们需要维护当前每个链表没有被合并的元素的最前面一个，kk 个链表就最多有 kk 个满足这样条件的元素，每次在这些元素里面选取 \textit{val}val 属性最小的元素合并到答案中。在选取最小元素的时候，我们可以用优先队列来优化这个过程。


这优先队列最好写的更清楚一些，为什么重载< 而不是> 要么写成

    struct Status {
        int val;
        ListNode *ptr;
        // 默认使用 less<>, 所以重载oprator <
        // 如果使用greater 则应该重载 >
        bool operator < (const Status &rhs) const {
            return val > rhs.val;
        }
    };
    // 定义 小顶堆
    priority_queue<Status, vector<Status>, less<Status> > q;
使用greater：

    struct Status {
        int val;
        ListNode *ptr;
       // 使用greater 则应该重载 >
        bool operator > (const Status &rhs) const {
            return val > rhs.val;
        }
    };
    // 定义 小顶堆
    priority_queue<Status, vector<Status>, greater<Status> > q;
    
    
    
### 代码
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:

    struct Status{
        int val;
        ListNode *ptr;
        // 默认使用 less<>, 所以重载oprator <
        bool operator < (const Status &p) const{
            return val > p.val;
        }
    };

    // 优先队列
    priority_queue <Status> q;

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        // 将 k 个链表的头结点加⼊最⼩堆
        for(auto list: lists){
            if(list)
                q.push({list->val, list});
        }

        ListNode phead , *tail =&phead;

        while (!q.empty()) {
            // 获取最⼩节点，接到结果链表中
            auto f = q.top();
            q.pop();
            tail->next = f.ptr;
            tail = tail->next;
            if (f.ptr->next) q.push({f.ptr->next->val, f.ptr->next});
        }
        return phead.next;
    }


};

复杂度分析

时间复杂度：考虑优先队列中的元素不超过 kk 个，那么插入和删除的时间代价为 O(\log k)O(logk)，这里最多有 knkn 个点，对于每个点都被插入删除各一次，故总的时间代价即渐进时间复杂度为 O(kn \times \log k)O(kn×logk)。
空间复杂度：这里用了优先队列，优先队列中的元素不超过 kk 个，故渐进空间复杂度为 O(k)O(k)。

```

## 剑指 Offer 22. 链表中倒数第k个节点

https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof


输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

### 思路
快慢指针的思想。

### code
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode* p = head;
        for(int i=0; i<k; i++){
            p = p->next;
        }
        ListNode* pHead = head;
        while(p){
            p=p->next;
            pHead=pHead->next;
        }
        return pHead;
    }
};
```
## 19. 删除链表的倒数第 N 个结点
https://leetcode.cn/problems/remove-nth-node-from-end-of-list/

### 思路
快慢指针

### code
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:

    ListNode* findFromEnd(ListNode*  head, int k){
        ListNode* p1 = head;
        for(int i=0; i<k; i++){
            p1 = p1->next;
        }
        ListNode* p2 = head;
        while(p1!=nullptr){
            p1 = p1->next;
            p2 = p2->next; 
        }
        return p2;
    }

    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;

        ListNode* p1 = head;
        for(int i=0; i<n; i++){
            p1 = p1->next;
        }
        ListNode* p2 = dummy;
        while(p1!=nullptr){
            p1 = p1->next;
            p2 = p2->next; 
        }

        p2->next = p2->next->next;
        return dummy->next;
    }
};
```


##  876. 链表的中间结点

https://leetcode.cn/problems/middle-of-the-linked-list

给定一个头结点为 head 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

```

示例 1：

输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.
示例 2：

输入：[1,2,3,4,5,6]
输出：此列表中的结点 4 (序列化形式：[4,5,6])
由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。
 

提示：

给定链表的结点数介于 1 和 100 之间。
```
### code
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast!=nullptr&& fast->next!=nullptr){
            fast = fast->next->next;
            slow = slow->next;
        }
        return slow;
    }
};
```

## 237. 删除链表中的节点
https://leetcode.cn/problems/delete-node-in-a-linked-list

请编写一个函数，用于 删除单链表中某个特定节点 。在设计函数时需要注意，你无法访问链表的头节点 head ，只能直接访问 要被删除的节点 。

题目数据保证需要删除的节点 不是末尾节点 。


```
class Solution {
public:
    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }
};
```

## 141. 环形链表

> 判断链表是否有环

https://leetcode.cn/problems/linked-list-cycle

给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false 。

### 思路
```
判断链表是否包含环
判断链表是否包含环属于经典问题了，解决⽅案也是⽤快慢指针：
每当慢指针 slow 前进⼀步，快指针 fast 就前进两步。
如果 fast 最终遇到空指针，说明链表中没有环；如果 fast 最终和 slow 相遇，那肯定是 fast 超过了
slow ⼀圈，说明链表中含有环。
```

### code
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast!=NULL && fast->next!=NULL){
            fast = fast->next->next;
            slow = slow->next;
            if (fast==slow)
                return true;
        }
        return false;
    }
};
```


## 面试题 02.08. 环路检测

https://leetcode.cn/problems/linked-list-cycle-lcci/

这个问题还有进阶版：如果链表中含有环，如何计算这个环的起点？

### 思路

我们假设快慢指针相遇时，慢指针 slow ⾛了 k 步，那么快指针 fast ⼀定⾛了 2k 步：

fast ⼀定⽐ slow 多⾛了 k 步，这多⾛的 k 步其实就是 fast 指针在环⾥转圈圈，所以 k 的值就是环⻓度
的「整数倍」。
假设相遇点距环的起点的距离为 m，那么结合上图的 slow 指针，环的起点距头结点 head 的距离为 k -
m，也就是说如果从 head 前进 k - m 步就能到达环起点。
巧的是，如果从相遇点继续前进 k - m 步，也恰好到达环起点。因为结合上图的 fast 指针，从相遇点开始
⾛k步可以转回到相遇点，那⾛ k - m 步肯定就⾛到环起点了：

### code
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast!=NULL && fast->next!=NULL){
            fast = fast->next->next;
            slow = slow->next;
            if (fast==slow)
                break;
        }
        if(fast==NULL || fast->next==NULL)
            return NULL;

        slow = head;
        while(fast!=slow){
            fast=fast->next;
            slow=slow->next;
        }

        return slow;
    }
};
```


## 160. 相交链表
https://leetcode.cn/problems/intersection-of-two-linked-lists

给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

图示两个链表在节点 c1 开始相交：

### 思路

我们可以让 p1 遍历完链表 A 之后开始遍历链表 B，让 p2 遍历完链表 B 之后开始遍历链表 A，这样相
当于「逻辑上」两条链表接在了⼀起。
如果这样进⾏拼接，就可以让 p1 和 p2 同时进⼊公共部分，也就是同时到达相交节点 c1

### code
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* p1 = headA;
        ListNode* p2 = headB;
        while(p1!=p2){
            if (p1==NULL) p1=headA;
            else p1=p1->next;
            if (p2==NULL) p2=headB;
            else p2=p2->next;
        }
        return p1;
    }
};
```
