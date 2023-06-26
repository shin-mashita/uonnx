#include "uonnx.h"

static int Reshape_init(Node * n)
{
	Tensor * x;
	Tensor * s;

	if((n->ninputs == 2) && (n->noutputs == 1))
	{
		x = n->inputs[0];
		s = n->inputs[1];
		
		if((x->ndim == 0) || (x->type == TENSOR_TYPE_UNDEFINED))
			return 0;
		if((s->ndim == 0) || (s->type != TENSOR_TYPE_INT64))
			return 0;
		return 1;
	}
	return 0;
}

static int Reshape_exit(Node * n)
{
	return 1;
}

static int Reshape_reshape(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x = n->inputs[0];
	Tensor * s = n->inputs[1];
	int64_t * ps = s->datas;
	int total_dim = 1;
	int total_shape = 1;
	int ndim = s->ndata;
	int dims[ndim];

	for(int i = 0; i < ndim; i++)
	{
		if(ps[i] == 0)
			dims[i] = x->dims[i];
		else if(ps[i] > 0)
			dims[i] = ps[i];
		else
		{
			for(int j = 0; j < x->ndim; j++)
				total_dim *= x->dims[j];
			for(int j = 0; j < ndim; j++)
			{
				if(ps[j] > 0)
					total_shape *= ps[j];
				else if(ps[j] == 0)
					total_shape *= x->dims[j];
			}
			dims[i] = total_dim / total_shape;
		}
	}
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void Reshape_operator(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x = n->inputs[0];
	char ** py = (char **)y->datas;
	char ** px = (char **)x->datas;

	if(x->type == TENSOR_TYPE_STRING)
	{
		for(size_t i = 0, l = y->ndata; i < l; i++)
		{
			if(py[i])
				free(py[i]);
			py[i] = strdup(px[i]);
		}
	}
	else
	{
		memcpy(y->datas, x->datas, x->ndata * onnx_tensor_type_sizeof(x->type));
	}
}

void resolver_default_op_Reshape(Node * n)
{
	if(n->opset >= 14)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BOOL:
		case TENSOR_TYPE_INT8:
		case TENSOR_TYPE_INT16:
		case TENSOR_TYPE_INT32:
		case TENSOR_TYPE_INT64:
		case TENSOR_TYPE_UINT8:
		case TENSOR_TYPE_UINT16:
		case TENSOR_TYPE_UINT32:
		case TENSOR_TYPE_UINT64:
		case TENSOR_TYPE_BFLOAT16:
		case TENSOR_TYPE_FLOAT16:
		case TENSOR_TYPE_FLOAT32:
		case TENSOR_TYPE_FLOAT64:
		case TENSOR_TYPE_COMPLEX64:
		case TENSOR_TYPE_COMPLEX128:
		case TENSOR_TYPE_STRING:
			n->init = Reshape_init;
			n->exit = Reshape_exit;
			n->reshape = Reshape_reshape;
			n->op = Reshape_operator;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BOOL:
		case TENSOR_TYPE_INT8:
		case TENSOR_TYPE_INT16:
		case TENSOR_TYPE_INT32:
		case TENSOR_TYPE_INT64:
		case TENSOR_TYPE_UINT8:
		case TENSOR_TYPE_UINT16:
		case TENSOR_TYPE_UINT32:
		case TENSOR_TYPE_UINT64:
		case TENSOR_TYPE_BFLOAT16:
		case TENSOR_TYPE_FLOAT16:
		case TENSOR_TYPE_FLOAT32:
		case TENSOR_TYPE_FLOAT64:
		case TENSOR_TYPE_COMPLEX64:
		case TENSOR_TYPE_COMPLEX128:
		case TENSOR_TYPE_STRING:
			n->init = Reshape_init;
			n->exit = Reshape_exit;
			n->reshape = Reshape_reshape;
			n->op = Reshape_operator;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 5)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BOOL:
		case TENSOR_TYPE_INT8:
		case TENSOR_TYPE_INT16:
		case TENSOR_TYPE_INT32:
		case TENSOR_TYPE_INT64:
		case TENSOR_TYPE_UINT8:
		case TENSOR_TYPE_UINT16:
		case TENSOR_TYPE_UINT32:
		case TENSOR_TYPE_UINT64:
		case TENSOR_TYPE_FLOAT16:
		case TENSOR_TYPE_FLOAT32:
		case TENSOR_TYPE_FLOAT64:
		case TENSOR_TYPE_COMPLEX64:
		case TENSOR_TYPE_COMPLEX128:
		case TENSOR_TYPE_STRING:
			n->init = Reshape_init;
			n->exit = Reshape_exit;
			n->reshape = Reshape_reshape;
			n->op = Reshape_operator;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
