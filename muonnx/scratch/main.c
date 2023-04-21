#include <uonnx.h>

Graph * graph_init(GraphProto gproto)
{
    Graph * g;

    g = malloc(sizeof(Graph));
    
}

TensorArena * arena_init(const int MAX_TENSORS, const int MAX_BLOCKS)
{
    int i;
    TensorArena * arena = malloc(sizeof(arena));

    if(arena)
    {
        arena->n_tensors = MAX_TENSORS;
        arena->tensors = malloc(sizeof(Tensor *)*MAX_TENSORS);
        arena->datas = malloc(sizeof(void *)*MAX_BLOCKS);
        memset(arena->datas, 0, sizeof(void *)*MAX_BLOCKS);
        
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
            if(!t->isProto)free(t->dims);
        }
        free(t);
    }
}

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
    

    // TODO: Add conditionals for NULL statements
    t->name = NULL;
    t->dims = NULL;
    t->ndim = NULL;

    t->name = strdup(name); // Free later
    t->type = type;

    //WORKAROUND FOR DOUBLE FREE
    t->isProto = 0;

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

    return t;

}

void tensor_apply(void * datas, size_t ndata, Tensor * t)
{
    if(t->name != NULL && t)
    {
        switch(t->type)
        {
            case TENSOR_TYPE_FLOAT32:
                memcpy(t->datas, datas, sizeof(float)*ndata);
                break;
                
            case TENSOR_TYPE_INT64:
                memcpy(t->datas, datas, sizeof(int64_t)*ndata);
                break;
        }
        
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

        // WORKAROUND FOR DOUBLE FREE
        t->dims = tp->dims;
        t->isProto = 1;
        // t->dims = malloc(sizeof(int64_t)*tp->n_dims); //
        // memcpy(t->dims, tp->dims, sizeof(int64_t)*tp->n_dims); // 
        
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
    int i; 

    ModelProto * model = NULL;
    GraphProto * graph = NULL;

    const char * filename = "./scratch/model.onnx";
    
    model = load_model(filename);

    TensorProto * tp0 = model->graph->initializer[0];
    TensorProto * tp1 = model->graph->initializer[1];
    TensorProto * tp2 = model->graph->initializer[2];

    TensorArena * arena = arena_init(10, 10000);

    void * testdata = malloc(sizeof(float)*5);
    float * p = (float *) testdata;
    for(int i=0; i<5; i++)
    {   
        p[i] = 0.6969 + i;
    }
    
    Tensor * t0 = tensor_init_from_proto(tp0, arena->tensors[0]);
    Tensor * t1 = tensor_init_from_proto(tp2, arena->tensors[2]);
    Tensor * t2 = tensor_init(tp2->name, tp2->data_type, tp2->dims, tp2->n_dims, arena, 3, 0*sizeof(float));
    // TODO: iteration is by bytes (data_idx = 1 -> 1 byte offset)
    // TODO: 

    Tensor * t3 = tensor_init(tp2->name, tp2->data_type, tp2->dims, tp2->n_dims, arena, 4, 10*sizeof(float));


    tensor_apply(tp2->float_data, tp2->n_float_data, t3);
    tensor_apply(testdata, 5, t2);

    dump_tensor(t1);
    dump_tensor(t3);
    
    for(int i = 0; i<10; i++)
    {
        if(arena->tensors[i]->name == NULL)
        {
            printf("%d is null\n",i);
        }
    }
    
    printf("\n\n\n");

    // uint16_t * buffer = malloc(sizeof(uint8_t)*20);
    // for(int i = 0; i< 20 ; i++)
    // {
    //     buffer[i] = i+2;
    // }

    // free(buffer);

    // for(int i = 0; i < 20; i++)
    // {
    //     printf("%d ", buffer[i]);
    // }

    free_model(model);
    free_arena(arena);


    float * v = (float *) arena->datas;
    for(int i=0; i<100; i++)
    {
        printf("%.2f ", v[i]);
    }

    
    return 0;
}