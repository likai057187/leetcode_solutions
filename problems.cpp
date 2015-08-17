#include"problems.h"

using namespace std;

 vector<int> twoSum(vector<int> &numbers, int target) {
	 map<int,int> mp;
	 map<int,int>::iterator it;
	 vector<int> ret;
	 for(int i=0;i<numbers.size();i++) {
		it = mp.find(target - numbers[i]);
		if(it!=mp.end()) {
			ret.push_back(it->second);
			ret.push_back(i+1);
			return ret;
		}
		it = mp.find(numbers[i]);
		if(it==mp.end()) {
			mp[numbers[i]] = i+1;
		}
	 }
	 
 }

int climbStairs(int n) {
	int * f = new int[n+1];
	f[0]=0;
	f[1]=1;
	f[2]=2;
	for(int i=3;i<n+1;i++) {
		f[i] = f[i-1] + f[i-2];
	}
	return f[n];
}

int reverse(int x) {
    int ret=0;
    while(x) {
        int temp = ret;
        ret = ret*10 + x%10;
        if(abs(temp)>abs(ret)) return 0;
        x/=10;
    }
    return ret;
}

ListNode *removeElements(ListNode *head, int val)
{
    ListNode **list = &head;

    while (*list != nullptr)
    {
        if ((*list)->val == val)
        {
            *list = (*list)->next;
        }
        else
        {
            list = &(*list)->next;
        }
    }

    return head;
}

class MinStack {
public:
	stack<int> stk,minStk;
    void push(int x) {
		stk.push(x);
		if(minStk.empty() || x<=minStk.top()) {
			minStk.push(x);
		}
    }

    void pop() {
		int top = stk.top();
		stk.pop();
		if(top == minStk.top()) {
			minStk.pop();
		}
    }

    int top() {
		return stk.top();
    }

    int getMin() {
		return minStk.top();
    }
};

string addBinary(string a, string b) {
    int carry=0;
    string result;
    for(int i=a.length()-1,j=b.length()-1;i>=0||j>=0;i--,j--) {
        int ai = i>=0?a[i]-'0':0;
        int bj = j>=0?b[j]-'0':0;
        int r = (ai+bj+carry)%2;
        carry = (ai+bj+carry)/2;
        result.insert(result.begin(),r+'0');
    }
    if(carry==1) result.insert(result.begin(),'1');
    return result;
}

uint32_t reverseBits(uint32_t n) {
	uint32_t temp = n | 1;
	return n>>1;
}

void rotate(vector<int>& nums, int k) {
	int n = nums.size();
	k = k % n;
	reverse(nums.begin(), nums.begin() + n);
    reverse(nums.begin(), nums.begin() + k);
    reverse(nums.begin() + k, nums.begin() + n);
}

int removeElement(vector<int>& nums, int val) {
	int i,j;
	for(i=0,j=0;j<nums.size();j++) {
		if(nums[j]!=val) {
			nums[i]=nums[j];
			i++;
		}
	}
	return i;
}

int removeDuplicates(vector<int>& nums) {
	int n = nums.size();
	if(n==0) return 0;
	int cur = 0;
	for(int i=1;i<n;i++) {
		if(nums[cur] != nums[i]) {
			nums[cur] = nums[i];
			cur++;
		}
	}
}

int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int bottomX = max(A,E);
        int bottomY = max(B,F);
        int topX = min(C,G);
        int topY = min(D,H);
        
    	int mXn; 
        if((topX<=0&&bottomX>=0) || (topY<=0&&bottomY>=0) || topX-bottomX<=0 || topY-bottomY<=0) {
    		mXn = 0;
    	} else {
    		mXn = (topX-bottomX)*(topY-bottomY);
    	}
    
    	return (C-A)*(D-B) + (G-E)*(H-F) - mXn;
}

bool containsNearbyDuplicate(vector<int>& nums, int k) {
    map<int,int> mp;
    for(int i=0;i<nums.size();i++) {
        if(mp.find(nums[i])==mp.end()) {
            mp[nums[i]]=i;
        } else {
            if(i-mp[nums[i]]<=k) {
                return true;
            } else {
				mp[nums[i]]=i;
			}
        }
    }
    return false;
}

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode * ret = NULL;
    ListNode ** pCur = &ret;
    while(l1 && l2) {
        if(l1->val<l2->val) {
            *pCur = l1;
            l1 = l1->next;
        } else {
            *pCur = l2;
            l2 = l2->next;
        }
        pCur = &((*pCur)->next);
    }
    l1->next==NULL?*pCur=l2:*pCur=l1;
    return ret;
}

bool findLong(vector<int>& v)
{
    if (v.empty()) return false;
    if (v.size() == 1) {
        cout << v[0] << endl;
        return true;
    }

	int longest_subarray_startIndex = v.size()-1;
	int longest_length = 1;
	int ascending_counter = 0;
	int descending_counter = 0;
	for(int i=v.size()-1;i!=0;i--){
		if(v[i-1] > v[i]) {
			descending_counter++; 
			ascending_counter=0;
		} else if(v[i-1] < v[i]) {
			ascending_counter++; 
			descending_counter=0;
		} else {
			ascending_counter++; 
			descending_counter++;
		}

		if(longest_length <= max(ascending_counter, descending_counter)) {
			longest_subarray_startIndex = i-1;
			longest_length = max(ascending_counter, descending_counter);
		}
	}

	//Print out results
	for(int i=longest_subarray_startIndex;i<=longest_subarray_startIndex+longest_length;i++) {
		cout<<v[i]<<" ";
	}
	cout<<endl;

	return true;
}

class NotEnoughElements {};
template <typename Container, typename Function>
typename Container::value_type Reduce(const Container& c, Function fn) throw (NotEnoughElements) {
	if(c.size()<=1) throw NotEnoughElements();
	typename Container::const_iterator it = c.begin();
	typename Container::value_type ret = *it;
	for(it=it+1;it!=c.end();it++) {
		ret = fn(ret,*it);
	}
	return ret;
}

//class A {
//public:
//	A() {};
//	~A() {cout<<"delete A\n";}
//	virtual void func() {cout<<"I am A\n";};
//};
//
//class B {
//public:
//	B() {};
//	virtual void func() {cout<<"I am B\n";};
//	~B() {cout<<"delete B\n";}
//};
//
//class C : public A {
//public:
//	C() {};
//	virtual void func() {cout<<"I am C\n";}
//	~C() {cout<<"delete C\n";}
//};

ListNode* swapPairs(ListNode* head) {
    if (head == NULL || head->next == NULL) {  
        return head;  
    }  
    ListNode* nextPair = head->next->next;  
    ListNode* newHead = head->next;  
    head->next->next = head;  
    head->next = swapPairs(nextPair);  
    return newHead; 
}

bool isAnagram(string s, string t) {
    if (s.length() != t.length()) return false;
    int n = s.length();
    map<char, int> counts;
    for (int i = 0; i < n; i++) {
        counts[s[i]]++;
        counts[t[i]]--;
    }
	map<char, int>::iterator it;
	for (it=counts.begin();it!=counts.end();it++) {
		if(it->second) {
			return false;
		}
	}
    return true;
}

vector<int> productExceptSelf(vector<int>& nums) {
	int n = nums.size();
	vector<int> ret(n,1);
	for(int i=0, prev=1;i<n;i++) {
		ret[i] = prev;
		prev*=nums[i];
	}
	for(int i=n-1,prev=1;i>=0;i--) {
		ret[i]*=prev;
		prev*=nums[i];
	}
	return ret;
}

int findMin(vector<int>& nums) {
	int start=0,end=nums.size()-1;
	int mid;
	while(start<end) {
		mid = start + (end-start)/2;
		if(nums[mid]<nums[end]) {
			end = mid;
		} else {
			start = mid+1;
		}
	}
	return nums[start];
}

int findPeakElement(vector<int>& nums) {
	if(nums.empty()) {
		return -1;
	}

	int start=0, end=nums.size()-1,mid=end/2;
	while(start+1<end) {
		mid=start+(end-start)/2;
		if(nums[mid]<nums[mid-1]) {
			end=mid;
		} else if(nums[mid]<nums[mid+1]) {
			start=mid;
		} else {
			return mid;
		}
	}
	mid=nums[start]>nums[end]?start:end;
	return mid;
}


ListNode *removeNthFromEnd(ListNode *head, int n) {  
	// Start typing your C/C++ solution below  
	// DO NOT write int main() function  
        ListNode *first=head, *second=head;
        int step=0;
        while(step<n && first!=NULL) {
            first=first->next;
            step++;
        }
        if(step==n && first==NULL) {
            head=head->next;
            delete second;
            return head;
        }
        while(first->next!=NULL) {
            first=first->next;
            second=second->next;
        }
        ListNode * temp = second->next;
        second->next = temp->next;
        delete temp;
        return head;
}

int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m,n);
	for(int i=0;i<m;i++) {
		dp[i][0]=1;
	}
	for(int i=0;i<n;i++) {
		dp[0][i]=1;
	}
	for(int i=1;i<m;i++) {
		for(int j=1;j<n;j++) {
			dp[i][j]=dp[i-1][j]+dp[i][j-1];
		}
	}
	return dp[m-1][n-1];
}

void setZeroes(vector<vector<int>>& matrix) {
    if(matrix.size()==0) return;
    int row = matrix.size();
    int col = matrix[0].size();
    for(int i=0;i<row;i++) {
        for(int j=0;j<col;j++) {
            if(matrix[i][j]==0) {
                int x=i, y=j;
                while(x>=0) {
                    matrix[x][j]=0;
                    x--;
                }
                while(y>=0) {
                    matrix[i][y]=0;
                    y--;
                }
            }
        }
    }
}

int rob(vector<int>& nums) {
           if(nums.empty()){
                return 0;
            }//if
            int size = nums.size();
            vector<int> dp(size,0);
            // dp[i]=max(num[i]+dp[i-2],dp[i-1])
            // dp[i]表示[0,i]取1个或多个不相邻数值的最大收益
            dp[0] = nums[0];
            dp[1] = max(nums[0],nums[1]);
            for(int i = 2;i < size;++i){
                dp[i] = max(dp[i-1],dp[i-2]+nums[i]);
            }//for
            return dp[size-1];
}

void reverseWords(string &s)  
{  
	int numOfWords=0;
    string rs;  
    for (int i = s.length()-1; i >= 0; )  
    {  
        while (i >= 0 && s[i] == ' ') i--;  
        if (i < 0) break;  
        if (!rs.empty()) rs.push_back(' ');  
        string t;  
        while (i >= 0 && s[i] != ' ') t.push_back(s[i--]);  
        reverse(t.begin(), t.end());  
        rs.append(t);  
		numOfWords++;
    }  
	if(numOfWords=0) s="";
	else s = rs;  
}

void DFS(vector<vector<int>>& ret, vector<int> curr, int n, int k, int level)
{
    if(curr.size() == k)
    {
        ret.push_back(curr);
        return;
    }
    if(curr.size() > k)  // consider this check to save run time
        return;

    for(int i = level; i <= n; ++i)
    {
        curr.push_back(i);
        DFS(ret,curr,n,k,i+1);
        curr.pop_back();
    }
}

vector<vector<int> > combine(int n, int k) {
    vector<vector<int>> ret;
    if(n <= 0) //corner case invalid check
        return ret;

    vector<int> curr;
    DFS(ret,curr, n, k, 1); //we pass ret as reference at here
    return ret;
}

vector<vector<int> > generate(int numRows) {
    vector<vector<int>> res;
    if(numRows<1) return res;
    vector<int> row(1,1);
    res.push_back(row);
        
    for(int i=2; i<=numRows; i++) {
        int prev = 1;
        for(int j=1; j<i-1; j++) {
            int temp = row[j];
            row[j] += prev;
            prev = temp;
        }
        row.push_back(1);
        res.push_back(row);
    }
    return res;
}

void swap(int& a1, int&a2){  
	int temp = a1;  
	a1=a2;  
	a2=temp;  
}

void rotate(vector<vector<int> > &matrix) {  
	int len = matrix[0].size();  
    for(int i =0; i<len-1; i++){  
		for(int j=0;j<len-i;j++){  
			swap(matrix[i][j], matrix[len-1-j][len-1-i]);  
		}  
	}  
	for(int i =0; i<len/2; i++){  
		for(int j=0;j<len;j++){  
			swap(matrix[i][j], matrix[len-i-1][j]);  
		}  
	}  
}

void deleteNode(ListNode* node) {
	if(node==NULL) return;
	node->val = node->next->val;
	ListNode * temp = node->next;
	if(temp->next) {
		node->next = temp->next;
	} else {
		node->next = NULL;
	}
	delete temp;
}

int titleToNumber(string s) {
    int ret=0;
    int i=0;
    for(;i<s.length();i++) {
        int n=(s[i]-'A')%26;
        ret+=n*26;
    }
    return ret+s[i]-'A'+1;
}

int searchInsert(vector<int>& nums, int target) {
    int left = 0, right = nums.size()-1, mid;
    while(left<right) {
        mid = left + (right-left+1)/2;
        if(nums[mid]==target) {
            return mid;
        } else {
            if(nums[mid]>target) {
                right = mid-1;
            } else {
                left = mid;
            }
        }
    }
    return nums[left]>=target?left:left+1;
}

ListNode* deleteDuplicates(ListNode* head) {
	ListNode * prev = head;
	ListNode * p = head->next;
	ListNode * temp;
	while(p!=NULL) {
		if(prev->val == p->val) {
			temp = p;
			p = p->next;
			prev->next = p;
			delete temp;
			continue;
		}
		p=p->next;
		prev=prev->next;
	}
	return head;
}

//int arr[] = {-2, -3, 4, -1, -2, 1, 5, -3};
int maxSubArray(vector<int>& nums) {
	if(nums.size()==0) return 0;
	int curMax = nums[0];
	int ret = nums[0];
	for(int i=1;i<nums.size();i++) {
		curMax = max(nums[i],curMax+nums[i]);
		ret = max(ret,curMax);
	}
	return ret;
}

int maxSubArrayPrint(vector<int> nums) {
	int ret = nums[0];
	int curMax = nums[0];
	int end=0;
	for(int i=1;i<nums.size();i++) {
		curMax = max(nums[i],curMax+nums[i]);
		if(curMax>ret) {
			ret=curMax;
			end=i;
		}
	}
	int cacheSum = ret;
	int start=end;
	while(cacheSum) {
		cacheSum -= nums[start];
		if(cacheSum) {
			start--;
		}
	}
	cout<<"Subarry: "<<start<<" - "<<end<<endl;
	return ret;
}

void MaximumSubArraySum(vector<int> input) {
    if (input.size() > 0) {
        int sum = 0;
        int MaxSum = 0;
        int start = 0;
        int end = 0;

        for (int i = 0; i < input.size(); i++) {
            sum += input[i];
            if (sum > 0) {
                if (sum > MaxSum) {
                    MaxSum = sum;
                    end = i;
                }
            } else {
                sum = 0; 
                start = i;
            }
        }

		cout << "Max Sum : " << MaxSum << " sub Array : " << start + 1 << " - " << end << endl;
    }
}

int romanToInt(string s) {
    map<char,int> mp;
    mp['I'] = 1, mp['V'] = 5, mp['X'] = 10, mp['L'] = 50, mp['C'] = 100, mp['D'] = 500, mp['M'] = 1000;
	int ret=0;
	for(int i=0;i<s.length()-1;i++) {
		if(mp[s[i]]>=mp[s[i+1]]) {
			ret+=mp[s[i]];
		} else {
			ret-=mp[s[i]];
		}
	}
	ret+=mp[s[s.length()-1]];
	return ret;
}

int findMin2(vector<int>& nums) {
    int left=0, right=nums.size()-1, mid;
	while(left<right) {
		mid = left + (right-left+1)/2;
		if(nums[mid]<nums[right]) {
			right = mid;
		} else {
			left = mid+1;
		}
	}
	return nums[left];
}

void sortColors(vector<int>& nums) {
	int redIndex = 0;
	int blueIndex = nums.size() -1;
	int i = 0;
	while(i<blueIndex+1) {
		if(nums[i]==0) {
			swap(nums[i],nums[redIndex]);
			redIndex++;
			i++;
			continue;
		}
		if(nums[i]==2) {
			swap(nums[i],nums[blueIndex]);
			blueIndex--;
			continue;
		}
		i++;
	}
}

int minPathSum(vector<vector<int>>& grid) {
	int row = grid.size();
	int col = grid[0].size();
	vector<vector<int>> dp(row,vector<int>(col,0));
	dp[0][0] = grid[0][0];
	for(int i=1;i<row;i++) {
		dp[i][0]=dp[i-1][0]+grid[i][0];
	}
	for(int i=1;i<col;i++) {
		dp[0][i]=dp[0][i-1]+grid[0][i];
	}
	for(int i=1;i<row;i++) {
		for(int j=1;j<col;j++) {
			dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j];
		}
	}
	return dp[row-1][col-1];
}

int ReverseInteger(int x) {
	int temp;
	int ret=0;;
	while(x) {
		temp = ret*10 + x%10;
		x/=10;
		if(temp/10 != ret) return 0;
		ret = temp;
	}
	return ret;
}