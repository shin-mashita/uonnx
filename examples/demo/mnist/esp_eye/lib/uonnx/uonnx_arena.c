#include "uonnx_arena.h"

TensorArena * arena_init(const int MAX_TENSORS, const size_t MAX_BYTES)
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
