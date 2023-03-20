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
    ONNX_LOG("mchar: %s\n", s->mchar);
    ONNX_LOG("num_in_mchar: %d\n", s->num_in_mchar);
    for(int i=0; i<10; i++)
    {
        ONNX_LOG("test_arr %d: %d\n", i, s->arr[i]->data);
    }

    free(s);
}

void testprint()
{
    ONNX_LOG("Hello uONNX\n");
}

int testsum(int a, int b)
{
    ONNX_LOG("Sum: %d\n", a+b);
    return a+b;
}