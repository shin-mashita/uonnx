#include <uonnx.h>

TensorArena * arena_malloc(const int MAX_TENSORS, const size_t MAX_BLOCKS) // TODO: Restrict by size not by number of tensors. Add param MAX_BYTES??.
{
	int i;
	Tensor * t;
	TensorArena * arena;

	arena = malloc(sizeof(TensorArena));
	if(arena)
	{
		arena->tensors = malloc(sizeof(Tensor *)*MAX_TENSORS);
		arena->datas = malloc(sizeof(void *)*MAX_BLOCKS);
		for(i = 0; i < MAX_TENSORS; i++)
		{
			arena->tensors[i] = malloc(sizeof(Tensor));
		}
		// Add here for floatdatas, intdatas etc 

		arena_init(arena, MAX_TENSORS, MAX_BLOCKS);
	}
	return arena;
}

void arena_init(TensorArena * arena, int n_tensors, size_t n_blocks)
{
	if(arena)
	{
		arena->n_tensors = n_tensors;
		
		memset(arena->datas, 0, sizeof(arena->datas)*n_blocks);
	}
}

void free_arena(TensorArena * arena)
{
	int i;
	Tensor * t;

	if(arena)
	{
		free(arena->tensors);
		free(arena->datas);

		for(i = 0; i < arena->n_tensors; i++)
		{
			arena->tensors[i] = malloc(sizeof(Tensor));
		}

		free(arena);
	}
}

void arena_add_tensor(TensorArena * arena)
{

}

Tensor * tensor_malloc(const char * name, TensorType type, int64_t * dims, size_t ndim, TensorArena * arena, int data_idx, int tensor_idx)
{
	// Can be NULL: dims, ndim, 
	Tensor * t = &arena->tensors[tensor_idx];

	if(t)
	{
		t->dims = malloc(sizeof(int64_t)*ndim);
		t->name = strdup(name); // Automatically assigns name. No need to initialize.
	}

	tensor_init(t, name, type, dims, ndim, arena, data_idx, tensor_idx);

	return t;
}

/* tensor_init: Initializes a tensor inside the arena. Also initializes datas inside arena*/
/* TODO: Change params from data_idx and tensor_idx to Planner * planner */
void tensor_init(Tensor * t, const char * name, TensorType type, int64_t * dims, size_t ndim, TensorArena * arena, int data_idx, int tensor_idx)
{
	int i, ndata = 1;

	if(ndim)
	{
		for(i=0; i < ndim; i++)
		{
			ndata *= dims[i];
		}
	}

	t->datas = &arena->datas[data_idx];
	printf("HERE ");
	addrcmp(t,t);
	t->type = type;
	t->ndim = ndim;
	memcpy(t->dims, dims, sizeof(int64_t)*ndim);
	if(ndata)
	{
		t->ndata = ndata;
	}

	switch(t->type)
	{
		case TENSOR_TYPE_FLOAT32:
			memset(t->datas, 0, sizeof(float)*ndata);
			break;
		case TENSOR_TYPE_INT64:
			memset(t->datas, 0, sizeof(int64_t)*ndata);
			break;
		default: // TODO: Add more dtypes.
			break;
	}
}



void free_tensor(Tensor * t)
{
	if(t)
	{
		if(t->name) free(t->name);
		if(t->dims) free(t->dims);
	}
}

// void set_tensor(); // Add values to tensor

int main()
{
	ModelProto * model = NULL;
	GraphProto * graph = NULL;

	const char * filename = "./scratch/model.onnx";

	model = load_model(filename);
	
	TensorProto * tp = model->graph->initializer[0];
	TensorProto * tp2 = model->graph->initializer[1];

	TensorArena * arena;
	arena = arena_malloc(10, 10000);

	Tensor * v, *v2;
	v = tensor_malloc(tp->name, tp->data_type, tp->dims, tp->n_dims, arena, 0, 0);



	v2 = tensor_malloc(tp2->name, tp2->data_type, tp2->dims, tp2->n_dims, arena, 3000, 1);

	addrcmp(v, &arena->tensors[0]);
	addrcmp(v->datas, &arena->datas[0]);
	addrcmp(v2, &arena->tensors[1]);
	addrcmp(v2->datas, &arena->datas[3000]);

	dump_tensor(v);
	dump_tensor(v2);

	// free_tensor(v);
	// free_tensor(v2);
	free_arena(arena);
	free_model(model);
	

    return 0;
}