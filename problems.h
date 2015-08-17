#include<iostream>
#include<vector>
#include<map>
#include<stack>
#include<stdint.h>
#include<algorithm>
#include<queue>
#include<set>

using namespace std;

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};

vector<int> twoSum(vector<int> &numbers, int target);
int climbStairs(int n);
int reverse(int x);
ListNode *removeElements(ListNode *head, int val);
class MinStack;
string addBinary(string a, string b);
uint32_t reverseBits(uint32_t n);
void rotate(vector<int>& nums, int k);
int removeElement(vector<int>& nums, int val);
int removeDuplicates(vector<int>& nums);
int computeArea(int A, int B, int C, int D, int E, int F, int G, int H);
bool containsNearbyDuplicate(vector<int>& nums, int k);
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
bool findLong(vector<int>& v);
ListNode* swapPairs(ListNode* head);
bool isAnagram(string s, string t);
vector<int> productExceptSelf(vector<int>& nums);
int findMin(vector<int>& nums);
int findPeakElement(vector<int>& nums);
ListNode *removeNthFromEnd(ListNode *head, int n);
int uniquePaths(int m, int n);
void setZeroes(vector<vector<int>>& matrix);
int rob(vector<int>& nums);
void reverseWords(string &s);
vector<vector<int> > combine(int n, int k);
vector<vector<int> > generate(int numRows);
void rotate(vector<vector<int> > &matrix);
void deleteNode(ListNode* node);
int titleToNumber(string s);
int searchInsert(vector<int>& nums, int target);
ListNode* deleteDuplicates(ListNode* head);
int maxSubArray(vector<int>& nums);
int maxSubArrayPrint(vector<int> nums);
void MaximumSubArraySum(vector<int> input);
int romanToInt(string s);
int findMin2(vector<int>& nums);
void sortColors(vector<int>& nums);
int minPathSum(vector<vector<int>>& grid);
int ReverseInteger(int x);