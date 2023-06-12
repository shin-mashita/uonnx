#include "uonnx_arena.h"

Tensor * arena_init_v2(const int MAX_TENSORS, const size_t MAX_BYTES)
{
    int i = 0, j = 0;
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

// free_arena_v2

TensorArena * arena_init(const int MAX_TENSORS, const size_t MAX_BYTES)
{
    int i = 0, j = 0;
    TensorArena * arena = (TensorArena *)malloc(sizeof(TensorArena));
    memset(arena, 0, sizeof(TensorArena));

    if(arena)
    {
        arena->tensors = (Tensor **)malloc(sizeof(Tensor *)*MAX_TENSORS);
        if(!arena->tensors)
        {
            free(arena);
            return NULL;
        }

        arena->datas = malloc(MAX_BYTES);
        if(!arena->datas)
        {
            free(arena->tensors);
            free(arena);
            return NULL;
        }
        memset(arena->datas, 0, MAX_BYTES);

        arena->n_bytes = MAX_BYTES;
        arena->n_tensors = MAX_TENSORS;

        for(i=0; i<MAX_TENSORS; i++)
        {
            arena->tensors[i] = malloc(sizeof(Tensor));
            if(arena->tensors[i])
            {
                arena->tensors[i]->name = NULL;
                arena->tensors[i]->dims = NULL;
                arena->tensors[i]->strides = NULL;
            }
            else
            {
                for(j = 0; j < MAX_TENSORS; j++)
                {
                    if(arena->tensors[j]) free(arena->tensors[j]);
                }
                free(arena->datas);
                free(arena->tensors);
                free(arena);
                return NULL;
            }
        }
        return arena;
    }

    return NULL;
}

TensorArena * arena_init_from_planner(Planner * planner)
{
    if(planner)
    {
        return arena_init(planner->max_arena_n_tensors, (size_t)planner->max_arena_size);
    }

    return NULL;
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
