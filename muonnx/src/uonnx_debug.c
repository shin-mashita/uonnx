#include "uonnx_debug.h"

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

void dump_graph(Graph * g)
{
    Node * n;

    int i, j;

    for(i = 0; i < g->nlen; i++)
    {
        n = &g->nodes[i];

        printf("Node %d\n",i);

        printf("Input\n");
        for(j = 0; j < n->ninputs; j++)
        {
            dump_tensor(n->inputs[j]);
        }

        printf("Output\n");
        for(j = 0; j < n->noutputs; j++)
        {
            dump_tensor(n->outputs[j]);
        }
    }
}

// for(i = 0; i < g->nlen; i++) // Add to uonnx_debug in dump_graph
// {
//     n = &g->nodes[i];
//     for(j = 0; j < gproto->node[i]->n_input; j++)
//     {
//         printf("\tInput %d: %s\n", j, gproto->node[i]->input[j]);
//     }
//     for(j = 0; j < gproto->node[i]->n_output; j++)
//     {
//         printf("\tOutput %d: %s\n", j, gproto->node[i]->output[j]);
//     }
// }

void dump_arena(TensorArena * arena, TensorType type, size_t n)
{
    int i;
    printf("Dumping Arena...\n");
    if(!arena)
    {
        printf("Arena is null");
        return;
    }

    for(i = 0; i < arena->n_tensors; i++)
    {
        printf("Tensor id %d\n",i);
        dump_tensor(arena->tensors[i]);
    }

    printf("\n\nArena Datas Overview for type %s\n", TensorType2String(type));
    switch(type)
    {
        case TENSOR_TYPE_FLOAT32:;
            float * v = (float *)arena->datas;
            for(i = 0; i < n; i++)
            {
                if(i%10 == 0) printf("\n");
                printf("%.2f ", v[i]);
            }
            break;
        default:
            break;
    }

    printf("\n");
}

void dump_tensor(Tensor * t)
{
    int i, j;

    if(!t || t->name == NULL)
    {
        printf("Tensor is null.\n");
        return;
    }
    
    printf("Tensor: \"%s\"\n",t->name);
    printf("\tdtype: %s\n", TensorType2String(t->type));
    printf("\tndata: %ld\n", t->ndata);
    printf("\tdatas: ");
    switch(t->type)
    {
        case TENSOR_TYPE_FLOAT32:
            for(i = 0; i < min(t->ndata, 10); i++)
            {
                float * v = (float *)t->datas;
                printf("%.3f ", v[i]);
            }
            break;
        case TENSOR_TYPE_INT64:
            for(i = 0; i < min(t->ndata, 10); i++)
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
    printf("\n");
    printf("\tisInitializer: %d", t->isInitializer);
    printf("\n\n");

    // printf("%s - %s (", v->name, TensorType2String(type));
    // for(i = 0; i < ndim; i++)
    // {
    //     printf("%d", dims[i]);
    //     if(i<ndim-1)printf(" x ");
    // }
    // printf(")\n");
}