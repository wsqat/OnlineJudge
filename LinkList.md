## 206. 反转链表
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
 

示例 1：


输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reverse-linked-list
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

> 思路

方法1.头插法：
思路：准备一个临时节点，然后遍历链表，准备两个指针head和next，每次循环到一个节点的时候，将head.next指向temp.next，并且将temp.next指向head，head和next向后移一位。
复杂度分析：时间复杂度：O(n)， n为链表节点数，空间复杂度：O(1)

作者：zz1998
链接：https://leetcode.cn/problems/reverse-linked-list/solution/dai-ma-jian-ji-yi-chong-huan-bu-cuo-de-j-lefc/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```
var reverseList = function (head) {
  let temp = new ListNode();
  let next = null;
  while (head) {
    next = head.next;//下一个节点
    head.next = temp.next;
    temp.next = head;//head接在temp的后面
    head = next;//head向后移动一位
  }
  return temp.next;
};

作者：zz1998
链接：https://leetcode.cn/problems/reverse-linked-list/solution/dai-ma-jian-ji-yi-chong-huan-bu-cuo-de-j-lefc/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```
```
public:
    ListNode* reverseList(ListNode* head) {
        //使用头插法
        ListNode* dummp = new ListNode(0);
        if(head == nullptr) return head;

        ListNode* cur = head, *p = dummp;
        while(cur){
            //从head摘下一个头
            ListNode* t = cur;
            cur = cur->next; //cur移到下一个
            t->next = p->next; //头插法插入
            p->next = t;

        }

        return dummp->next;

    }
};
```

方法2. 迭代

```
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre=nullptr, *cur=head, *nxt=head;
        while(cur){
            nxt = cur->next;
            //逐个节点翻转
            cur->next=pre;
            //更新指针位置
            pre=cur;
            cur=nxt;
        }
        // 返回反转后的头结点
        return pre;
    }
};
```

方法3.递归：
思路：用递归函数不断传入head.next，直到head==null或者heade.next==null，到了递归最后一层的时候，让后面一个节点指向前一个节点，然后让前一个节点的next置为空，直到到达第一层，就是链表的第一个节点，每一层都返回最后一个节点。
复杂度分析：时间复杂度：O(n)，n是链表的长度。空间复杂度：O(n)， n是递归的深度，递归占用栈空间，可能会达到n层

作者：zz1998
链接：https://leetcode.cn/problems/reverse-linked-list/solution/dai-ma-jian-ji-yi-chong-huan-bu-cuo-de-j-lefc/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        // 想象递归已经层层返回，到了最后一步
        // 以链表 1->2->3->4->5 为例，现在链表变成了 5->4->3->2->null，1->2->null
        // 此时 newHead是5，head是1
        ListNode newHead = reverseList(head.next);
        // 最后的操作是把链表 1->2->null 变成 2->1->null
        // head是1，head.next是2，head.next.next = head 就是2指向1，此时链表为 2->1->2，是一个环
        head.next.next = head;
        // 防止链表循环，1指向null，此时链表为 2->1->null，整个链表为 5->4->3->2->1->null
        head.next = null;
        return newHead;
    }
```


## 25. K 个一组翻转链表

给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

 

示例 1：


输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/reverse-nodes-in-k-group
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


https://mp.weixin.qq.com/s/A-dQ9spsP_Iu1Y4iCRP9nA
> 思路

「反转以 a 为头结点的链表」其实就是「反转 a 到 null 之间的结点」，那么如果让你「反转 a 到 b 之间的结点」，你会不会？

只要更改函数签名，并把上面的代码中 null 改成 b 即可：

首先，我们要实现一个 reverse 函数反转一个区间之内的元素。在此之前我们再简化一下，给定链表头结点，如何反转整个链表？
````
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre=nullptr, *cur=head, *nxt=head;
        while(cur){
            nxt = cur->next;
            //逐个节点翻转
            cur->next=pre;
            //更新指针位置
            pre=cur;
            cur=nxt;
        }
        // 返回反转后的头结点
        return pre;
    }
};
```
这次使用迭代思路来实现的，借助动画理解应该很容易。

```
class Solution {
public:

    /** 反转区间 [a, b) 的元素，注意是左闭右开 */
    ListNode* reverse(ListNode* head, ListNode* tail){
        ListNode *pre=nullptr,*cur=head,*nxt=head;
        // while 终止的条件改一下就行了
        while(cur!=tail){
            nxt = cur->next;
            cur->next = pre;
            pre = cur;
            cur = nxt;
        }
        // 返回反转后的头结点
        return pre;
    }

    ListNode* reverseKGroup(ListNode* head, int k) {
        if (!head) return head;
        // 区间 [a, b) 包含 k 个待反转元素
        ListNode *a=head, *b=head;
        for(int i=0;i<k;i++){
            // 不足 k 个，不需要反转，base case
            if (!b) return head;
            b = b->next;
        }
        // 反转前 k 个元素
        ListNode *newHead = reverse(a, b);
        // 递归反转后续链表并连接起来
        a->next = reverseKGroup(b, k);
        return newHead;
    }
};
```
