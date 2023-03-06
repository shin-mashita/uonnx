// TEST on how to make a module with a file.c and a file.h

#include "mtest.h"

Teststruct * init_teststruct(Teststruct * s)
{
    srand(time(NULL));

    s = malloc(sizeof(Teststruct));
    s->mchar = "THIS IS A TEST STRING";
    s->num_in_mchar = strlen(s->mchar);
    s->arr = malloc(10*sizeof(Testarr *));

    for(int i=0; i<10; i++)
    {
        s->arr[i] = malloc(sizeof(Testarr));
    }

    for(int i=0; i<10; i++)
    {
        s->arr[i]->data = rand()%10;
    }

    return s;
}

void print_teststruct(Teststruct * s)
{
    printf("mchar: %s\n", s->mchar);
    printf("num_in_mchar: %d\n", s->num_in_mchar);
    for(int i=0; i<10; i++)
    {
        printf("test_arr %d: %d\n", i, s->arr[i]->data);
    }
}

void testprint()
{
    printf("Hello uONNX\n");
}

int testsum(int a, int b)
{
    printf("Sum: %d\n", a+b);
    return a+b;
}