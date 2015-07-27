#include <stdio.h> 
#include <stdlib.h> 
  
void showheading(); 
int hivalue(int stu[4][5]); 
int lovalue(int stu[4][5]); 
void displaymenu(){ 
printf("====================================================\n"); 
printf("                         MENU                                  \n"); 
printf("=====================================================\n"); 
printf("     1.View all student records\n"); 
printf("     2.View a student records by ID\n"); 
printf("     3.Show the highest and the lowest scores\n"); 
  
     } 
void viewall(int stu[4][5]){ 
 int i,j; 
//display heading 
showheading(); 
 for(i=0;i<4;i++){ 
   for(j=0;j<5;j++) printf("%s",stu[i][j]);printf("\t\t"); 
   printf("\n"); 
     } 
} 
void viewbyid(int stu[4][5]){ 
     int id,i,j; 
     int l=0; 
     printf("Please enter a student's ID:"); 
     scanf("%d",&id); 
     for(i=0;i<4;i++){ 
      if(stu[i][0]==id){ 
                        showheading();l=1; 
                        for(j=0;j<5;j++)printf("%d",stu[i][j]);printf("\t\t");} 
       printf("\n");} 
      if(l==0) printf("Not found!\n"); 
     
     } 
void showhl(int stu[4][5]){ 
     printf("The higest final score is:%d",hivalue(stu)); 
     printf("\n"); 
     printf("The lowest final score is:%d",lovalue(stu)); 
     printf("\n"); 
     
     } 
void showheading(){ 
printf("=====================================================\n"); 
printf("StudentID      Quiz1          Quiz2          Mid-term         Final\n"); 
printf("=====================================================\n"); 
     } 
int hivalue(int stu[4][5]){ 
    int max,i; 
    max=stu[0][4]; 
    for(i=0;i<4;i++) 
      if(max<stu[i][4]) max=stu[i][4]; 
    return(max); 
    
} 
int lovalue(int stu[4][5]){ 
    int min,i; 
    min=stu[0][4]; 
    for(i=0;i<4;i++) 
      if(min>stu[i][4]) min=stu[i][4]; 
    return(min); 
    
} 
     
int main(int argc, char *argv[]) 
{ 
//construct 2d array to store students'records 
int stu[4][5]={{1232,32,34,43,43},{2345,34,34,54,35},{3432,45,54,56,34},{3456,56,34,34,56}}; 
  
//show menu 
  displaymenu(); 
  int yourchoice; 
  char confirm; 
  do 
  { 
    printf("Enter your choice(1-3):"); 
    scanf("%d",&yourchoice); 
  
  switch(yourchoice){ 
    case 1:viewall(stu);break; 
    case 2:viewbyid(stu);break; 
    case 3:showhl(stu);break; 
    default:printf("invalid"); 
                   } 
                   
      printf("Press y or Y to continue:"); 
       scanf("%s",confirm); 
    }while(confirm=='y'||confirm=='Y'); 
  
  system("PAUSE"); 
  
  return EXIT_SUCCESS; 
} 