/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_FLOOR

static int Floor_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Floor_exit(Node * n)
{
	return 1;
}

static int Floor_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Floor_bfloat16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		py[i] = float32_to_bfloat16(floorf(v));
	}
}

static void Floor_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(floorf(v));
	}
}

static void Floor_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = floorf(px[i]);
}

static void Floor_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = floor(px[i]);
}

void resolver_default_op_Floor(Node * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BFLOAT16:
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_float64;
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
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_float64;
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
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Floor_init;
			n->exit = Floor_exit;
			n->reshape = Floor_reshape;
			n->op = Floor_float64;
			break;
		default:
			break;
		}
	}
}

#endif