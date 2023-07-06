/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_CEIL

static int Ceil_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Ceil_exit(Node * n)
{
	return 1;
}

static int Ceil_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Ceil_bfloat16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		py[i] = float32_to_float16(ceilf(v));
	}
}

static void Ceil_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(ceilf(v));
	}
}

static void Ceil_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ceilf(px[i]);
}

static void Ceil_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = ceil(px[i]);
}

void resolver_default_op_Ceil(Node * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BFLOAT16:
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_float64;
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
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_float64;
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
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Ceil_init;
			n->exit = Ceil_exit;
			n->reshape = Ceil_reshape;
			n->op = Ceil_float64;
			break;
		default:
			break;
		}
	}
}

#endif