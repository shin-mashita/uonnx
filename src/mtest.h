#ifndef __MTEST_H__
#define __MTEST_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

typedef struct testarr
{
    int data;
}Testarr;

typedef struct teststruct
{
    char * mchar;
    Testarr ** arr;
    int num_in_mchar;
}Teststruct;

Teststruct * init_teststruct(struct teststruct * s);
void print_teststruct(struct teststruct * s);
void testprint();
int testsum(int a, int b);



#endif