#include "uonnx.h"

#ifdef UONNX_OPS_PRELU

static int PRelu_init(Node * n)
{
	if((n->ninputs == 2) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int PRelu_exit(Node * n)
{
	return 1;
}

static int PRelu_reshape(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];

	return onnx_tensor_reshape_identity(y, a, a->type);
}

static void PRelu_int32(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	int32_t * py = (int32_t *)y->datas;
	int32_t * pa = (int32_t *)a->datas;;
	int32_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		if(pa[i] < 0)
		{
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_int64(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	int64_t * py = (int64_t *)y->datas;
	int64_t * pa = (int64_t *)a->datas;;
	int64_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		if(pa[i] < 0)
		{
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_uint32(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * pa = (uint32_t *)a->datas;;
	uint32_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		if(pa[i] < 0)
		{
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_uint64(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * pa = (uint64_t *)a->datas;;
	uint64_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		if(pa[i] < 0)
		{
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_float16(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa = (uint16_t *)a->datas;;
	uint16_t * pb;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(pa[i]);
		if(v < 0)
		{
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = float32_to_float16(v * float16_to_float32(*pb));
		}
		else
			py[i] = float32_to_float16(v);
	}
}

static void PRelu_float32(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	float * py = (float *)y->datas;
	float * pa = (float *)a->datas;;
	float * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		if(pa[i] < 0)
		{
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

static void PRelu_float64(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	double * py = (double *)y->datas;
	double * pa = (double *)a->datas;;
	double * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		if(pa[i] < 0)
		{
			pb = onnx_tensor_broadcast_map_address(b, y, i);
			py[i] = pa[i] * (*pb);
		}
		else
			py[i] = pa[i];
	}
}

void resolver_default_op_PRelu(Node * n)
{
	if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_INT32:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_int32;
			break;
		case TENSOR_TYPE_INT64:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_int64;
			break;
		case TENSOR_TYPE_UINT32:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_uint32;
			break;
		case TENSOR_TYPE_UINT64:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_uint64;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = PRelu_init;
			n->exit = PRelu_exit;
			n->reshape = PRelu_reshape;
			n->op = PRelu_float64;
			break;
		default:
			break;
		}
	}
}

#endif