#include <uonnx.h>


TensorArena * arena_init(const int MAX_TENSORS, const int MAX_BYTES)
{
    int i;
    TensorArena * arena = malloc(sizeof(arena));

    if(arena)
    {
        arena->n_tensors = MAX_TENSORS;
        arena->tensors = malloc(sizeof(Tensor *)*MAX_TENSORS);
        arena->datas = malloc(sizeof(void *)*MAX_BYTES);
        memset(arena->datas, 0, sizeof(void *)*MAX_BYTES);
        
        for(i=0; i<arena->n_tensors; i++)
        {
            arena->tensors[i] = malloc(sizeof(Tensor));
            arena->tensors[i]->name = NULL;
        }

        return arena;
    }

    return NULL;
}

// Tensor * tensor_malloc()
// {

// }

Tensor * tensor_init(   const char * name, 
                        TensorType type, 
                        int64_t * dims, 
                        int ndim, 
                        TensorArena * arena, 
                        int tensor_idx, 
                        int data_idx)
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
    

    t->name = strdup(name); // Free later
    t->type = type;

    t->dims = malloc(sizeof(int64_t)*ndim); //Free later
    memcpy(t->dims, dims, sizeof(int64_t)*ndim);

    t->ndim = ndim;

    for(i=0; i<ndim; i++)
    {
        ndata*= dims[i];
    }

    if(ndata != 0)
    {
        t->ndata = ndata;
    }

    t->datas = &arena->datas[data_idx];

    if(t->datas)
    {
        memset(t->datas, 0, sizeof(void *)*ndata);
    }

    return t;

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
        t->dims = tp->dims;
        t->ndim = tp->n_dims;

        for(i < 0; i < t->ndim; i++)
        {
            ndata *= t->dims[i];
        }

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

int main()
{
    ModelProto * model = NULL;
    GraphProto * graph = NULL;

    const char * filename = "./scratch/model.onnx";
    
    model = load_model(filename);

    TensorProto * tp0 = model->graph->initializer[0];
    TensorProto * tp1 = model->graph->initializer[1];

    TensorArena * arena = arena_init(10, 10000);

    void * testdata = malloc(sizeof(float)*5);
    float * p = (float *) testdata;
    for(int i=0; i<5; i++)
    {   
        p[i] = 0.6969 + i;
    }

    Tensor * t0 = tensor_init_from_proto(tp0, arena->tensors[1]);
    Tensor * t1 = tensor_init_from_proto(tp1, arena->tensors[5]);
    
    
    Tensor * t2 = tensor_init(tp0->name, tp0->data_type, tp0->dims, tp0->n_dims, arena, 1, 0);

    
    for(int i = 0; i<10; i++)
    {
        if(arena->tensors[i]->name == NULL)
        {
            printf("%d is null\n",i);
        }
    }

    dump_tensor(t0);
    dump_tensor(t1);

    dump_tensor(t2);

    free_model(model);
    
    return 0;
}