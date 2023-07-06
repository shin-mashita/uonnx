/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_GLOBALMAXPOOL

static int GlobalMaxPool_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int GlobalMaxPool_exit(Node * n)
{
	return 1;
}

static int GlobalMaxPool_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int ndim = x->ndim;
	int dims[ndim];
	int i;

	for(i = 0; i < ndim; i++)
	{
		if(i < 2)
			dims[i] = x->dims[i];
		else
			dims[i] = 1;
	}
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void GlobalMaxPool_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < C; ++j)
		{
			o = i * C + j;
			v = float16_to_float32(px[o * m]);
			for(k = 1; k < m; ++k)
				v = fmaxf(v, float16_to_float32(px[o * m + k]));
			py[o] = float32_to_float16(v);
		}
	}
}

static void GlobalMaxPool_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < C; ++j)
		{
			o = i * C + j;
			py[o] = px[o * m];
			for(k = 1; k < m; ++k)
				py[o] = fmaxf(py[o], px[o * m + k]);
		}
	}
}

static void GlobalMaxPool_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < C; ++j)
		{
			o = i * C + j;
			py[o] = px[o * m];
			for(k = 1; k < m; ++k)
				py[o] = fmax(py[o], px[o * m + k]);
		}
	}
}

void resolver_default_op_GlobalMaxPool(Node * n)
{
	if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = GlobalMaxPool_init;
			n->exit = GlobalMaxPool_exit;
			n->reshape = GlobalMaxPool_reshape;
			n->op = GlobalMaxPool_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = GlobalMaxPool_init;
			n->exit = GlobalMaxPool_exit;
			n->reshape = GlobalMaxPool_reshape;
			n->op = GlobalMaxPool_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = GlobalMaxPool_init;
			n->exit = GlobalMaxPool_exit;
			n->reshape = GlobalMaxPool_reshape;
			n->op = GlobalMaxPool_float64;
			break;
		default:
			break;
		}
	}
}

#endif