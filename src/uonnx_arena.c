/**
 * @file uonnx_arena.c
 * @author Cedric Encarnacion (dakarashin0@gmail.com)
 * @brief 
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "uonnx_arena.h"

TensorArena * arena_init(const int MAX_TENSORS, const size_t MAX_BYTES)
{
    TensorArena * arena = (TensorArena *)malloc(sizeof(TensorArena));

    if(!arena)
        return NULL;

    memset(arena, 0, sizeof(TensorArena));

    arena->tensors = (Tensor **)malloc(sizeof(Tensor *)*MAX_TENSORS);
    arena->datas = malloc(MAX_BYTES);
    if(!arena->tensors || !arena->datas)
    {
        if(arena->tensors)
            free(arena->tensors);
        if(arena->datas)
            free(arena->datas);
        free(arena);
        return NULL;
    }

    memset(arena->datas, 0, MAX_BYTES);

    arena->n_bytes = MAX_BYTES;
    arena->n_tensors = 0;
    arena->MAX_TENSORS = MAX_TENSORS;

    return arena;
}

void free_arena(TensorArena * arena)
{
    int i = 0;

    if(arena)
    {
        for(i=0; i<arena->n_tensors; i++)
        {
            if(arena->tensors[i])
            {
                free_tensor_from_arena(arena->tensors[i]);
            }
        }

        if(arena->datas)
        {
            free(arena->datas);
        }

        if(arena->tensors)
        {
            free(arena->tensors);
        }
        
        free(arena);
    }
}


void arena_add_tensor(Tensor * tensor, TensorArena * arena, int arena_pos)
{
    if(!tensor || !arena || arena->n_tensors == arena->MAX_TENSORS)
        return;

    if(arena_pos >= 0)
    {
        tensor->datas = arena->datas + arena_pos;
    }
    arena->tensors[arena->n_tensors] = tensor;
    arena->n_tensors ++;
}

void arena_add_initializer(TensorProto * initializer, TensorArena * arena)
{
    int i = 0;
    Tensor * t;
    size_t ndata = 1;

    if(!initializer || !arena || arena->n_tensors == arena->MAX_TENSORS)
        return;

    t = tensor_alloc_nodatas(shash(initializer->name), initializer->data_type, (int *)initializer->dims, initializer->n_dims, 1);

    t->dims = malloc(sizeof(int) * initializer->n_dims);
    t->strides = malloc(sizeof(int) * initializer->n_dims);

    if (t->dims && t->strides)
    {
        t->strides[initializer->n_dims - 1] = 1;

        for (i = initializer->n_dims - 2; i >= 0; i--)
        {
            t->strides[i] = initializer->dims[i + 1] * t->strides[i + 1];
        }

        for (i = 0; i < initializer->n_dims; i++)
        {
            t->dims[i] = initializer->dims[i];
        }

        t->ndim = initializer->n_dims;
    }

    for (i = 0; i < t->ndim; i++)
    {
        ndata *= t->dims[i];
    }

    if(initializer->raw_data.len > 0 && initializer->raw_data.data)
    {
        switch(t->type)
        {
            case TENSOR_TYPE_FLOAT32:
            case TENSOR_TYPE_INT64:
                {
                    t->datas = initializer->raw_data.data;
                    t->ndata = initializer->raw_data.len/(onnx_tensor_type_sizeof(t->type));
                }
                break;
            default:
                break;
        }
    }
    else
    {
        switch (t->type)
        {
            case TENSOR_TYPE_FLOAT32:
            {
                t->datas = initializer->float_data;
                t->ndata = initializer->n_float_data;
            }
                break;

            case TENSOR_TYPE_INT64:
                t->datas = initializer->int64_data;
                t->ndata = initializer->n_int64_data;
                break;

            case TENSOR_TYPE_FLOAT16:
            {
                // TODO: Allocation handling scheme for float16 tensors
                t->datas = initializer->int32_data;
                t->ndata = ndata;
            }
                break;

            default:
                break; // TODO: ADD SUPPORT FOR OTHER DTYPES
        }
    }


    if (ndata != t->ndata)
    {
        free_tensor_from_arena(t);
        return;
    }

    arena_add_tensor(t, arena, -1);
}

void arena_add_intermediate(PlanProto * plan, TensorArena * arena)
{
    Tensor * t;

    if(!plan || !arena || arena->n_tensors == arena->MAX_TENSORS)
        return;

    t = tensor_alloc_nodatas(plan->id, plan->type, plan->dims, plan->n_dims, 0);

    arena_add_tensor(t, arena, plan->start_idx);
}


Tensor * tensor_search(TensorArena * arena, const char * name)
{
    int i = 0;

    for(i = 0; i < arena -> n_tensors; i++)
    {
        if(shash(name)==arena->tensors[i]->id)
        {
            return arena->tensors[i];
        }
    }
    
    return NULL;
}
