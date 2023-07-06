/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_TAN

static int Tan_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Tan_exit(Node * n)
{
	return 1;
}

static int Tan_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Tan_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(tanf(v));
	}
}

static void Tan_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = tanf(px[i]);
}

static void Tan_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = tan(px[i]);
}

void resolver_default_op_Tan(Node * n)
{
	if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Tan_init;
			n->exit = Tan_exit;
			n->reshape = Tan_reshape;
			n->op = Tan_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Tan_init;
			n->exit = Tan_exit;
			n->reshape = Tan_reshape;
			n->op = Tan_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Tan_init;
			n->exit = Tan_exit;
			n->reshape = Tan_reshape;
			n->op = Tan_float64;
			break;
		default:
			break;
		}
	}
}

#endif