#include "uonnx_allocator.h"

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
    t->strides = NULL;
    t->ndim = 0;

    t->name = strdup(name); // Free later
    t->type = type;

    //WORKAROUND FOR DOUBLE FREE
    t->isInitializer = 0;

    t->strides = malloc(sizeof(int)*ndim);
    t->dims = malloc(sizeof(int)*ndim);


    
    if(t->dims && t->strides)
    {
        t->strides[ndim - 1] = 1;
        
        for(i = ndim - 2; i >= 0; i--)
        {
            t->strides[i] = dims[i + 1] * t->strides[i + 1];
        }

        for(i = 0; i < ndim; i++)
        {
            t->dims[i] =  dims[i];
        }

        // memcpy(t->dims, dims, sizeof(int) * ndim); // Also works
        t->ndim = ndim;
    }
    else
    {
        if(t->strides)
        {
            free(t->strides);
            t->strides = NULL;
        }
        if(t->dims)
        {
            free(t->dims);
            t->dims = NULL;
        }
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
        t->datas = arena->datas + data_idx; // EDITED from &arena->datas[data_idx] to arena->datas + data_idx
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

    return NULL;
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
        t->strides = malloc(sizeof(int)*tp->n_dims);

        if(t->dims && t->strides)
        {
            t->strides[tp->n_dims - 1] = 1;
            
            for(i = tp->n_dims - 2; i >= 0; i--)
            {
                t->strides[i] = tp->dims[i + 1] * t->strides[i + 1];
            }

            for(i = 0; i < tp->n_dims; i++)
            {
                t->dims[i] =  tp->dims[i];
            }

            t->ndim = tp->n_dims;
        }

        for(i = 0; i < t->ndim; i++)
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


Graph * graph_init(GraphProto * gproto, TensorArena * arena, Planner * planner)
{
    int i = 0, j = 0, isInitializer = 0;
    int n_tensor_count = 0;
    Graph * g;
    Node * n;
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
        if(!tensor_search(arena, vp->name))
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
                    tensor_init_from_proto(gproto->initializer[j], arena->tensors[n_tensor_count]);
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
                tensor_init_from_value_info(vp, arena, n_tensor_count, get_plan(vp->name, planner));
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
            tensor_init_from_value_info(vp, arena, n_tensor_count, get_plan(vp->name, planner));
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
            tensor_init_from_value_info(vp, arena, n_tensor_count, get_plan(vp->name, planner));
            n_tensor_count++;
        }
    }

    for(i = 0; i < g->nlen; i++)
    {
        n = &g->nodes[i];
        n->proto = gproto->node[i];

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

        // if(!n->operator)
        // {
        //     resolver_solve_operator(&resolver_default, n);
        // }
    }

    return g;


}