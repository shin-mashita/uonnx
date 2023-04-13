#include <uonnx.h>

TensorArena * arena_init(int max_tensors, const size_t MAX_BLOCKS) // TODO: Restrict by size not by number of tensors. Add param MAX_BYTES??.
{
	int i;
	Tensor * t;

	TensorArena * arena;
	arena = malloc(sizeof(TensorArena));
	if(arena)
	{
		arena->tensors = malloc(sizeof(Tensor *)*max_tensors);
		for(i = 0; i < max_tensors; i++)
		{
			t = &arena->tensors[i];
			t = malloc(sizeof(Tensor));
			t = NULL;
		}

		arena->n_tensors = max_tensors;
		arena->datas = malloc(sizeof(void *)*MAX_BLOCKS);
		memset(arena->datas, 0, sizeof(arena->datas)*MAX_BLOCKS);
	}
	return arena;
}

void free_arena(TensorArena * arena)
{
	if(arena)
	{
		free(arena->tensors);
		free(arena->datas);
		free(arena);
	}
}

void arena_add_tensor(TensorArena * arena)
{

}


Tensor * tensor_init(const char * name, TensorType type, int64_t * dims, size_t ndim, TensorArena * arena, int data_idx, int tensor_idx)
{
	int i, ndata = 1;
	
	if(ndim)
	{
		for(i=0; i < ndim; i++)
		{
			ndata *= dims[i];
		}
	}
	
	Tensor * t = &arena->tensors[tensor_idx];
	// Tensor * t = malloc(sizeof(Tensor));

	if(t)
	{
		// Allocate memory for Tensor name, dims

		t->dims = malloc(sizeof(int64_t)*ndim);
		t->datas = &arena->datas[data_idx]; // TODO: Change idx based on planner
		// TODO: Separate dataarena for different dtypes? 

		// t->datas = malloc(sizeof(float)*ndata);
		// addrcmp(t->datas,t);

		// Populate assigned/initial values for t

		t->name = strdup(name);
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

	return t;
}

void free_tensor(Tensor * t)
{
	if(t)
	{
		if(t->name) free(t->name);
		if(t->dims) free(t->dims);
		// if(t->datas) free(t->datas);

		free(t);
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
	arena = arena_init(10, 10000);

	Tensor * v, *v2;
	v = tensor_init(tp->name, tp->data_type, tp->dims, tp->n_dims, arena, 0, 0);
	dump_tensor(v);
	addrcmp(v, &arena->tensors[0]);
	addrcmp(v->datas, &arena->datas[0]);

	v2 = tensor_init(tp2->name, tp2->data_type, tp2->dims, tp2->n_dims, arena, 3000, 1);
	dump_tensor(v2);

	addrcmp(v2, &arena->tensors[1]);
	addrcmp(v2->datas, &arena->datas[3000]);

	// free_tensor(v);
	// free_tensor(v2);
	// free_arena(arena);
	free_model(model);
	

    return 0;
}