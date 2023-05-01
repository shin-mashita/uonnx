#include <uonnx.h>
#include "model.h"
// TODO: Tensor apply inconsistents with sizeof() and with actual indexes

Plan * plan_create(char * tensor_name, size_t idx)
{
    Plan * p = malloc(sizeof(Plan));

    if(p)
    {
        p->tensor_name = NULL;
        p->index = NULL;

        if(tensor_name && idx)
        {
            p->tensor_name = strdup(tensor_name);
            p->index = idx;
        }

        return p;
    }

    return NULL;
}

void free_plan(Plan * p)
{
    if(p)
    {
        if(p->tensor_name) free(p->tensor_name);
        free(p);
    }
}

Planner * planner_init(int n_plans, size_t max_bytes, int max_tensors)
{
    int i = 0;
    Planner * planner = malloc(sizeof(Planner));

    if(!planner)
    {
        return NULL;
    }

    planner->plans = malloc(sizeof(Plan *)*n_plans);
    if(!planner->plans)
    {
        free(planner);
        return NULL;
    }

    for(i = 0; i < n_plans; i++)
    {
        planner->plans[i] = plan_create(NULL, NULL);
    }

    planner->n_plans = n_plans;
    planner->max_arena_size = max_bytes;
    planner->max_arena_n_tensors = max_tensors;

    return planner;
}

void free_planner(Planner * planner)
{
    int i = 0;
    Plan * p;

    if(planner)
    {
        if(planner->plans)
        {
            for(i = 0; i < planner->n_plans; i++)
            {
                p = planner->plans[i];
                free_plan(p);
            }
            
            free(planner->plans);
        }

        free(planner);
    }
}

void planner_add(char * tensor_name, size_t idx,Planner * planner)
{
    int i = 0;

    for(i = 0; i < planner->n_plans; i++)
    {
        if(planner->plans[i]->tensor_name == NULL)
        {
            planner->plans[i]->tensor_name = strdup(tensor_name);
            planner->plans[i]->index = idx;

            return;
        }
    }

    printf("Planner full%d\n");
}

size_t get_plan(char * tensor_name, Planner * planner)
{
    int i = 0;

    for(i = 0; i < planner->n_plans; i++)
    {
        if(strcmp(planner->plans[i]->tensor_name, tensor_name)==0)
        {
            return planner->plans[i]->index;
        }
    }

    return -1;
}

void dump_planner(Planner * planner)
{
    int i = 0;
    for(i = 0; i < planner->n_plans; i++)
    {
        if(planner->plans[i]->tensor_name)
        {
            printf("Tensor \"%s\" assigned in idx %d\n", planner->plans[i]->tensor_name, planner->plans[i]->index);
        }
        else
        {
            printf("Tensor NULL assigned in idx NULL\n");
        }
    }
}

// Add planner_init_from_GraphProto()

Tensor * tensor_search(TensorArena * arena, char * name)
{
    int i = 0;

    for(i = 0; i < arena->n_tensors; i++)
    {
        if(arena->tensors[i]->name && strcmp(arena->tensors[i]->name, name)==0)
        {
            return arena->tensors[i];
        }
    }
    
    return NULL;
}

TensorArena * arena_init(const int MAX_TENSORS, const int MAX_BYTES)
{
    int i;
    TensorArena * arena = malloc(sizeof(arena));

    if(arena)
    {
        arena->n_tensors = MAX_TENSORS;
        arena->tensors = malloc(sizeof(Tensor *)*MAX_TENSORS);
        arena->datas = malloc(MAX_BYTES);
        arena->n_bytes = MAX_BYTES;
        if(!arena->datas)
        {
            free(arena->tensors);
            free(arena);
            return NULL;
        }
        memset(arena->datas, 0, MAX_BYTES);
        
        for(i=0; i<arena->n_tensors; i++)
        {
            arena->tensors[i] = malloc(sizeof(Tensor));
            arena->tensors[i]->name = NULL;
        }

        return arena;
    }

    return NULL;
}

void free_arena(TensorArena * arena)
{
    int i;

    if(arena)
    {
        if(arena->datas) // BUG: CAUSING DOUBLE FREE!!!
        {
            free(arena->datas);
        }

        for(i=0; i<arena->n_tensors; i++)
        {
            free_tensor(arena->tensors[i]);
        }

        if(arena->tensors)
            free(arena->tensors);

        free(arena);
    }
}

void free_tensor(Tensor * t)
{
    if(t)
    {
        if(t->name)
        {
            free(t->name);
            free(t->dims);
        }
        free(t);
    }
}

/* 
 * TENSOR_INIT 
 * Initialize a tensor with datas memset'd to 0
 * If arena == NULL, malloc instead
 * Required params (name, type, dims , ndim)
 * Return address pointing to tensor
 */
Tensor * tensor_init(   const char * name, 
                        TensorType type, 
                        int * dims,  // TODO: Check if function can accept int64_t for 32 bit MCUs
                        int ndim, 
                        TensorArena * arena, 
                        int tensor_idx, 
                        int data_idx) // byte offset. REQUIRED: multiply by sizeof(dtype)
{
    // Check if & needed
    int i = 0; size_t ndata = 1;
    Tensor * t;
    
    if(arena != NULL)
    {
        t = arena->tensors[tensor_idx];
    }
    else
    {
        
        t = malloc(sizeof(Tensor)); //TODO: Find way to free if arena == NULL
    }
    

    // TODO: Add conditionals for NULL statements
    t->name = NULL;
    t->dims = NULL;
    t->ndim = NULL;

    t->name = strdup(name); // Free later
    t->type = type;

    //WORKAROUND FOR DOUBLE FREE
    t->isInitializer = 0;

    t->dims = malloc(sizeof(int)*ndim);
    t->ndim = ndim;

    // memcpy(t->dims, dims, sizeof(int) * ndim);
    for(i = 0; i < t->ndim; i++)
    {
        t->dims[i] =  dims[i];
    }

    for(i=0; i<ndim; i++)
    {
        ndata*= t->dims[i];
    }

    if(ndata != 0)
    {
        t->ndata = ndata;
    }

    if(arena)
    {
        t->datas = &arena->datas[data_idx];
    }
    else
    {
        switch(t->type)
        {
            case TENSOR_TYPE_FLOAT32:
                t->datas = malloc(sizeof(float)*ndata);
                memset(t->datas, 0, sizeof(float)*ndata);
                break;
            case TENSOR_TYPE_INT64:
                t->datas = malloc(sizeof(int64_t)*ndata);
                memset(t->datas, 0, sizeof(int64_t)*ndata);
                break;
            default:
                break;
        }
    }

    

    return t;

}


// TODO add params for tensor arena and for planner
Tensor * tensor_init_from_value_info(ValueInfoProto * v, TensorArena * arena, int tensor_idx, int data_idx)
{
    Tensor * t; 
    int type;
    int * dims = NULL;
    int ndim;
    int i;

    if(!v || !v->name)
    {
        return NULL;
    }

    if(v->type->value_case == ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE)
    {
        type = v->type->tensor_type->elem_type;
        ndim = v->type->tensor_type->shape->n_dim;

        if(ndim>0)
        {
            dims = malloc(sizeof(int)*ndim);
            if(dims)
            {
                for(i = 0; i < ndim; i++)
                {
                    if(v->type->tensor_type->shape->dim[i]->value_case == ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE)
                    {
                        dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
                    }
                    else
                    {
                        dims[i] = 1;
                    }
                }
            }
        }


        t = tensor_init(v->name, type, dims, ndim, arena, tensor_idx, data_idx);

        if(dims)
        {
            free(dims);
            dims = NULL;
        }

        return t;
    }
}

void tensor_apply(void * datas, size_t size, Tensor * t)
{
    if(t->name != NULL && t)
    {
        memcpy(t->datas, datas, size); // Run with sizeof()
        // switch(t->type)
        // {
        //     case TENSOR_TYPE_FLOAT32:
        //         memcpy(t->datas, datas, ndata);
        //         break;
                
        //     case TENSOR_TYPE_INT64:
        //         memcpy(t->datas, datas, ndata);
        //         break;
        // }
        
    }
}


// Use TensorProto addresses for permanent tensor (weights)
// Does not memcpy
Tensor * tensor_init_from_proto(TensorProto * tp, Tensor * tensor) 
{
    int i;
    size_t ndata = 1;
    Tensor * t;
    
    if(tensor == NULL)
    {
        t = malloc(sizeof(Tensor));
    }
    else
    {
        t = tensor;
    }
    
    if(tp)
    {
        t->name = tp->name;
        t->type = tp->data_type;

        t->dims = malloc(sizeof(int)*tp->n_dims); 
        for(i = 0; i < tp->n_dims; i++)
        {
            t->dims[i] = tp->dims[i];
        }
        
        t->ndim = tp->n_dims;

        for(i < 0; i < t->ndim; i++)
        {
            ndata *= t->dims[i];
        }

        t->isInitializer = 1;

        switch(t->type)
        {
            case TENSOR_TYPE_FLOAT32:
                t->datas = tp->float_data;
                t->ndata = tp->n_float_data;
                break;
            
            case TENSOR_TYPE_INT64:
                t->datas = tp->int64_data;
                t->ndata = tp->n_int64_data;
                break;
            
            default:
                break; // ADD SUPPORT FOR OTHER DTYPES
        }


        if(ndata != t->ndata) 
        {
            return NULL;
        }


        return t;
    }

    return NULL;
}


Graph * graph_init(GraphProto * gproto, TensorArena * arena, Planner * planner)
{
    int i = 0, j = 0, isInitializer = 0;
    int n_tensor_count = 0;
    Graph * g;
    Node * n;
    Tensor * t;
    ValueInfoProto * vp;

    g = malloc(sizeof(Graph));

    if(!gproto) return NULL;
    if(!g) return NULL;
    
    memset(g, 0, sizeof(Graph));

    g->nlen = gproto->n_node;

    g->nodes = malloc(sizeof(Node)*g->nlen);
    if(!g->nodes)
    {
        free(g);
        return NULL;
    }

    for(i = 0; i < gproto->n_input; i++)
    {
        vp = gproto->input[i];
        if(!tensor_search(arena, vp->name));
        {
            for(j = 0; j < gproto->n_initializer; j++)
            {
                if(strcmp(gproto->initializer[j]->name, vp->name)==0)
                {
                    if(n_tensor_count >= arena->n_tensors)
                    {
                        printf("Tensor blocks not enough. Expand arena.\n");
                        return NULL;
                    }
                    t = tensor_init_from_proto(gproto->initializer[j], arena->tensors[n_tensor_count]);
                    n_tensor_count++;
                    isInitializer = 1;
                }
            }

            if(!isInitializer)
            {
                if(n_tensor_count >= arena->n_tensors)
                {
                    printf("Tensor blocks not enough. Expand arena.\n");
                    return NULL;
                }
                t = tensor_init_from_value_info(vp, arena, n_tensor_count, get_plan(vp->name, planner));
                n_tensor_count++;
            }
            isInitializer = 0;
        }
    }

    for(i = 0; i < gproto->n_output; i++)
    {
        vp = gproto->output[i];
        if(!tensor_search(arena, vp->name))
        {
            if(n_tensor_count >= arena->n_tensors)
            {
                printf("Tensor blocks not enough. Expand arena.\n");
                return NULL;
            }
            t = tensor_init_from_value_info(vp, arena, n_tensor_count, get_plan(vp->name, planner));
            n_tensor_count++;
        }
    }

    for(i = 0; i < gproto->n_value_info; i++)
    {
        vp = gproto->value_info[i];
        if(!tensor_search(arena, vp->name))
        {
            if(n_tensor_count >= arena->n_tensors)
            {
                printf("Tensor blocks not enough. Expand arena.\n");
                return NULL;
            }
            t = tensor_init_from_value_info(vp, arena, n_tensor_count, get_plan(vp->name, planner));
            n_tensor_count++;
        }
    }

    for(i = 0; i < g->nlen; i++)
    {
        n = &g->nodes[i];

        n->inputs = malloc(sizeof(Tensor *)*gproto->node[i]->n_input);
        if(n->inputs)
        {
            n->ninputs = gproto->node[i]->n_input;
            for(j = 0; j < gproto->node[i]->n_input; j++)
            {
                n->inputs[j] = tensor_search(arena, gproto->node[i]->input[j]);
            }
        }

        n->outputs = malloc(sizeof(Tensor *)*gproto->node[i]->n_output);

        if(n->outputs)
        {
            n->noutputs = gproto->node[i]->n_output;
            for(j = 0; j < gproto->node[i]->n_output; j++)
            {
                n->outputs[j] = tensor_search(arena, gproto->node[i]->output[j]);
            }
        }
    }

    return g;


}

int main()
{
    int i = 0, j = 0; 

    ModelProto * model = NULL;
    GraphProto * graph = NULL;
    Tensor * tv0;

    const char * filename = "./scratch/model.onnx";
    
    model = load_model(filename);

    ValueInfoProto * vp0 = model->graph->input[0];
    TensorProto * tp0 = model->graph->initializer[0];

    Planner * planner = planner_init(13, sizeof(float)*(2560+6272+6272+100), 21);
    planner_add("Parameter193_reshape1", 0, planner); // fp32 (2560) - Node 0 10
    planner_add("Input3", sizeof(float)*(2560+6272), planner); // fp32 (784) - Node 1
    planner_add("Convolution28_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 1 2 
    planner_add("Plus30_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (6272) - Node 2 3 
    planner_add("ReLU32_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 3 4
    planner_add("Pooling66_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (1568) - Node 4 5
    planner_add("Convolution110_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 5 6
    planner_add("Plus112_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (3136) - Node 6 7
    planner_add("ReLU114_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 7 8
    planner_add("Pooling160_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (256) - Node 8 9 
    planner_add("Pooling160_Output_0_reshape0", sizeof(float)*(2560), planner); // fp32 (256) - 9 10
    planner_add("Times212_Output_0", sizeof(float)*(2560+256), planner); // fp32 (10) - Node 10 11
    planner_add("Plus214_Output_0", sizeof(float)*(0), planner); // fp32 (10) - Node 11


    TensorArena * arena = arena_init(planner->max_arena_n_tensors, planner->max_arena_size); // Initialization of arena


    // NEEDED: Convert int64_t dims first into int dims
    int * t0dims = malloc(sizeof(int)*tp0->n_dims);
    for(i = 0; i < tp0->n_dims; i++)
    {
        t0dims[i] = tp0->dims[i];
    }

    // Tensor * t0 = tensor_init(tp0->name, (TensorType)tp0->data_type, t0dims, tp0->n_dims, arena, 0, 1*sizeof(float));
    // dump_tensor(t0);

    // free(t0dims); // NEEDED

    // Tensor * t1 = tensor_init_from_value_info(vp0, arena, 9, 4*sizeof(float));

    // static const float testdata0[] = {1,2,3,4,5};
    // static const float testdata1[] = {6,7,8,9,10};

    // Tensor * test = tensor_search(arena, "Input3");
    // tensor_apply(testdata0, sizeof(testdata0), test);

    // test = tensor_search(arena, "Parameter193");
    // tensor_apply(testdata1, sizeof(testdata1), test);


    Graph * g = graph_init(model->graph, arena, planner);


    dump_graph(g);

    Tensor * t = tensor_search(arena, "Input3");
    tensor_apply((void*)input_3, sizeof(input_3), t);

    Tensor * t0 = tensor_search(arena, "Input3");
    Tensor * t1 = tensor_search(arena, "Pooling66_Output_0");
    

    // printf("%d\n", arena->n_bytes);
    // dump_planner(planner);
    free_planner(planner);


    // dump_arena(arena, TENSOR_TYPE_FLOAT32, 100);

    // for(i = 0; i < model->graph->n_input; i++)
    // {
    //     printf("%s\n", model->graph->input[i]->name);
    // }
    // printf("\n\n");
    // for(i = 0; i < model->graph->n_output; i++)
    // {
    //     printf("%s\n", model->graph->output[i]->name);
    // }
    // printf("\n\n");
    // for(i = 0; i < model->graph->n_value_info; i++)
    // {
    //     printf("%s\n", model->graph->value_info[i]->name);
    // }
    // printf("\n\n");
    // for(i = 0; i < model->graph->n_initializer; i++)
    // {
    //     printf("%s\n", model->graph->initializer[i]->name);
    // }


    return 0;
}
