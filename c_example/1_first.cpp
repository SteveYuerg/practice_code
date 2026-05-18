#include <stdio.h>
#include <iostream>
#include <stdlib.h>
 
int tt(int* a){
   std::cout << "tt: " << *a << std::endl;
   std::cout << "tt: " << a << std::endl;
   return *(a + 3);
}

void minmax(int arr[], int len, int* min, int* max){
   for (int i=0; i < len; i++){
      if (arr[i] < *min){
         *min = arr[i];
      }
      else if (arr[i] > *max){
         *max = arr[i];
      }
   }
   
}

void swap(void* a, void* b, int len){
   char* p1 = (char*)a;
   char* p2 = (char*)b;
   char temp = 0;

   for (int i=0; i < len; i++){
      temp = *p1;
      *p1 = *p2;
      *p2 = temp;
      p1++;
      p2++;
   }
}
 
int main ()
{
   int  var[10] = {1,4,14,0,5,6,7,-3,9,-1};   /* 实际变量的声明 */
   int  *ip = &var[0];        /* 指针变量的声明 */
   int *ip2;       /* 另一个指针变量的声明 */
   float b1 = 3.14f;
   double b2 = 3.14;
   long b3 = 123456789L;
   long long b4 = 123456789012345LL;

   int min = var[0];
   int max = var[0];

   int arr[3] = {10, 20, 30};

 
//    ip = &var[0];  /* 在指针变量中存储 var 的地址 */
//    ip2 = &ip;
 
   printf("var 变量的地址: %p\n", &var  );
 
   /* 在指针变量中存储的地址 */
   printf("ip 变量存储的地址: %p\n", ip );
 
   /* 使用指针访问值 */
   printf("*ip 变量的值: %d\n", *ip );

   // 打印指针变量 ip 本身的地址
   printf("ip 变量的地址: %p\n", &ip );
   
   printf("浮点数: %f, %zu\n", b1, sizeof(b1));
   printf("double浮点数: %f, %zu\n", b2, sizeof(b2));
   printf("long整数: %ld, %zu\n", b3, sizeof(b3));
   printf("long long整数: %lld, %zu\n", b4, sizeof(b4));

   printf("\n");

   printf("数组sizeof，字节大小: %zu\n", sizeof(arr));
   printf("数组长度，元素个数: %zu\n", sizeof(arr) / sizeof(arr[0]));

   printf("\n");
//    printf("ip2 变量存储的地址: %p\n", ip2);
//    printf("ip2 变量的值: %d\n", *ip2);
   std::cout << *(ip + 1) << std::endl;
   std::cout << "888 " << 223 << ":\n";
   int b = tt(ip);
   std::cout << "b: " << b << std::endl;

   std::cout << "--------------------------" << std::endl;
   minmax(var, 10, &min, &max);
   std::cout << "Min: " << min << ", Max: " << max << std::endl;

   int sa = 100;
   int sb = 200;

   std::cout << "sa:"  << &sa << "," << sa << std::endl;
   std::cout << "sb:"  << &sb << "," << sb << std::endl;
   swap(&sa, &sb, sizeof(int));
   std::cout << "after swap:" << sa << "," << sb << std::endl;

   std::cout << "sa:"  << &sa << "," << sa << std::endl;
   std::cout << "sb:"  << &sb << "," << sb << std::endl;
 
   return 0;
}