#include "problems.h"

int main(void) {

	int arr[] = {-2, -3, 4, -1, -2, 1, 5, -3};
	vector<int> vInput (arr, arr + sizeof(arr) / sizeof(int) );

	ListNode * n1 = new ListNode(1);
	ListNode * n2 = new ListNode(2);
	ListNode * n3 = new ListNode(3);
	ListNode * n4 = new ListNode(4);
	n1->next=n2;
	n2->next=n3;
	n3->next=n4;
	n4->next=NULL;

	system("pause");
	return 0;
}