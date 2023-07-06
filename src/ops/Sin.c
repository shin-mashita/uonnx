/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_SIN

static int Sin_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Sin_exit(Node * n)
{
	return 1;
}

static int Sin_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Sin_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(sinf(v));
	}
}

static void Sin_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = sinf(px[i]);
}

static void Sin_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = sin(px[i]);
}

void resolver_default_op_Sin(Node * n)
{
	if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Sin_init;
			n->exit = Sin_exit;
			n->reshape = Sin_reshape;
			n->op = Sin_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Sin_init;
			n->exit = Sin_exit;
			n->reshape = Sin_reshape;
			n->op = Sin_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Sin_init;
			n->exit = Sin_exit;
			n->reshape = Sin_reshape;
			n->op = Sin_float64;
			break;
		default:
			break;
		}
	}
}

#endif
