#include <stdio.h> 
#include <stdlib.h> 
int main() 
{
  int array[10]={};
    int i, j, temp;
    printf("please input 10 numbers");
    for(i=0;i<10;i++)
    {
     scanf("%d", &j);
     array[i]=j;
    } 
  for (i=0; i<9; i++)  
  {
    for (j=i+1; j<10; j++)
      if (array[i] < array[j])
        {
          temp = array[i];
          array[i] = array[j];
          array[j] = temp;
        }

  }
  for (i=0; i<9; i++)  
  {
    printf("%d",array[i]);
  } 
}