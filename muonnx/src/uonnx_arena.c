#include "uonnx_arena.h"

TensorArena * arena_init(const int MAX_TENSORS, const size_t MAX_BYTES)
{
    int i;
    TensorArena * arena = (TensorArena *)malloc(sizeof(arena));

    if(arena)
    {
        arena->n_tensors = MAX_TENSORS;
        arena->tensors = (Tensor **)malloc(sizeof(Tensor *)*MAX_TENSORS);
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
            arena->tensors[i] = (Tensor *)malloc(sizeof(Tensor));
            arena->tensors[i]->name = NULL;
        }

        return arena;
    }

    return NULL;
}

TensorArena * arena_init_from_planner(Planner * planner)
{
    if(planner)
    {
        return arena_init(planner->max_arena_n_tensors, planner->max_arena_size);
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
