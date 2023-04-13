#include "uonnx_debug.h"

int i, j;

void addrcmp(void * ptr1, void * ptr2)
{
    if(ptr1 == ptr2)
    {
        printf("Same address: %p\n", ptr1);
    }
    else
    {
        printf("Different addresses: addr1 = %p, addr2 = %p \n", ptr1, ptr2);
    }
}

void dump_tensor(Tensor * t)
{
    printf("Tensor: \"%s\"\n",t->name);
    printf("\tdtype: %s\n", TensorType2String(t->type));
    printf("\tndata: %ld\n", t->ndata);
    printf("\tdatas: ");
    switch(t->type)
    {
        case TENSOR_TYPE_FLOAT32:
            for(i = 0; i < 10; i++)
            {
                float * v = (float *)t->datas;
                printf("%f ", v[i]);
            }
            break;
        case TENSOR_TYPE_INT64:
            for(i = 0; i < 10; i++)
            {
                int64_t * v = (int64_t *)t->datas;
                printf("%ld ", v[i]);
            }
            break;
        default: // TODO: Log for other dtypes
            break;
    }
    printf("... \n");
    printf("\tndim: %d\n", t->ndim);
    printf("\tdims: ");
    for(i = 0; i < t->ndim; i++) printf("%d ", t->dims[i]);
    printf("\n\n");
}