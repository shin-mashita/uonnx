#include "uonnx_debug.h"
 
void addrcmp(void *ptr1, void *ptr2)
{
    if (ptr1 == ptr2)
    {
        printf("Same address: %p\n", ptr1);
    }
    else
    {
        printf("Different addresses: addr1 = %p, addr2 = %p \n", ptr1, ptr2);
    }
}

void get_cpu_heap(const char * TAG)
{
   struct mallinfo info;
   memset(&info,0,sizeof(struct mallinfo));
   info = mallinfo();
   printf("CPU Heap at [%s]: uordblks: %d | hblkhd: %d | total: \n", TAG, info.uordblks, info.hblkhd, info.uordblks + info.hblkhd);
}

void dump_plannerproto(PlannerProto * planner)
{
    int i = 0, j = 0;

    printf("Arena max tensors: %d\n", planner->arena->max_ntensors);
    printf("Arena max bytes: %d\n", planner->arena->max_bytes);
    
    for(i = 0; i < planner->n_plans; i++)
    {
        printf("Planner: %x\n", planner->plans[i]->id);
        printf("\tstart_idx: %d\n", planner->plans[i]->start_idx);
        printf("\tndims: %d\n", planner->plans[i]->n_dims);
        printf("\tdims: ");
        for(j = 0; j < planner->plans[i]->n_dims; j++)
        {
            printf("%d ", planner->plans[i]->dims[j]);
        }
        printf("\n");
    }
}

void dump_node(Node * n)
{
    int j;
    printf("Operator: %s\n", n->proto->op_type);

    printf("Input\n");
    for (j = 0; j < n->ninputs; j++)
    {
        dump_tensor(n->inputs[j]);
    }

    printf("Output\n");
    for (j = 0; j < n->noutputs; j++)
    {
        dump_tensor(n->outputs[j]);
    }
}


void dump_graph(Graph *g)
{
    Node *n;

    int i;

    for (i = 0; i < g->nlen; i++)
    {
        n = &g->nodes[i];

        printf("Node %d\n", i);
        dump_node(n);
    }
}

void dump_arena(TensorArena *arena, TensorType type, size_t n)
{
    int i;
    printf("Dumping Arena...\n");
    if (!arena)
    {
        printf("Arena is null");
        return;
    }

    for (i = 0; i < arena->n_tensors; i++)
    {
        printf("Tensor id %d\n", i);
        dump_tensor(arena->tensors[i]);
    }

    printf("\n\nArena Datas Overview for type %s\n", TensorType2String(type));
    switch (type)
    {
    case TENSOR_TYPE_FLOAT32:;
        float *v = (float *)arena->datas;
        for (i = 0; i < n; i++)
        {
            if (i % 10 == 0)
                printf("\n");
            printf("%.2f ", v[i]);
        }
        break;
    default:
        break;
    }

    printf("\n");
}

void dump_tensor(Tensor *t)
{
    int i;

    if (!t)
    {
        printf("Tensor is null.\n");
        return;
    }

    printf("Tensor: \"%x\"\n", t->id);
    printf("\tdtype: %s\n", TensorType2String(t->type));
    printf("\tndata: %d\n", t->ndata);
    printf("\tdatas: ");
    switch (t->type)
    {
    case TENSOR_TYPE_FLOAT32:
        for (i = 0; i < min(t->ndata, 10); i++)
        {
            float *v = (float *)t->datas;
            printf("%.3f ", v[i]);
        }
        break;
    case TENSOR_TYPE_INT64:
        for (i = 0; i < min(t->ndata, 10); i++)
        {
            int64_t *v = (int64_t *)t->datas;
            printf("%lld ", v[i]);
        }
        break;
    default: // TODO: Log for other dtypes
        break;
    }
    printf("... \n");
    printf("\tndim: %d\n", t->ndim);
    printf("\tdims: ");
    for (i = 0; i < t->ndim; i++)
        printf("%d ", t->dims[i]);
    printf("\n");
    printf("\tstrides: ");
    for (i = 0; i < t->ndim; i++)
        printf("%d ", t->strides[i]);
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

void dump_planner(Planner *planner)
{
    int i = 0;
    for (i = 0; i < planner->n_plans; i++)
    {
        if (planner->plans[i]->tensor_name)
        {
            printf("Tensor \"%s\" assigned in idx %d\n", planner->plans[i]->tensor_name, planner->plans[i]->index);
        }
        else
        {
            printf("Tensor NULL assigned in idx NULL\n");
        }
    }
}

// Add to uonnx_tests ?

// void op_testrun_no_attr(Tensor **inputs, Tensor **outputs, int ninputs, int noutputs, char *op_type, int opset)
// {
//     int i = 0;
//     Node *n = (Node *)malloc(sizeof(Node));
//     if (!n)
//         return;

//     n->ninputs = ninputs;
//     n->noutputs = noutputs;
//     n->opset = opset;

//     n->inputs = inputs;
//     n->outputs = outputs;

//     if (n->inputs && n->outputs)
//     {
//         resolver_solve_operator(&resolver_default, n, op_type);
//         if (n->operator)
//         {
//             printf("Input tensors for optype %s\n", op_type);
//             for (i = 0; i < ninputs; i++)
//             {
//                 dump_tensor(n->inputs[i]);
//             }

//             n->operator(n);

//             printf("Output tensors for optype %s\n", op_type);
//             for (i = 0; i < noutputs; i++)
//             {
//                 dump_tensor(n->outputs[i]);
//             }
//         }
//     }

//     if (n)
//     {
//         if (n->inputs)
//             free(n->inputs);
//         if (n->outputs)
//             free(n->outputs);
//         free(n);
//     }
// }

// void op_test_Add(void *in_a,
//                  TensorType type_in_a,
//                  int *dims_in_a,
//                  size_t ndims_in_a,
//                  void *in_b, 
//                  void *output, 
//                  TensorType type_b)
// {
//     int ninputs = 2;
//     int noutput = 1;
//     int i = 0;
//     Tensor *t;
//     t->tensor_init("Test1", )
// }