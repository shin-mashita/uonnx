/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_ASIN
static int Asin_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Asin_exit(Node * n)
{
	return 1;
}

static int Asin_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Asin_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(asinf(v));
	}
}

static void Asin_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = asinf(px[i]);
}

static void Asin_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = asin(px[i]);
}

void resolver_default_op_Asin(Node * n)
{
	if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Asin_init;
			n->exit = Asin_exit;
			n->reshape = Asin_reshape;
			n->op = Asin_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Asin_init;
			n->exit = Asin_exit;
			n->reshape = Asin_reshape;
			n->op = Asin_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Asin_init;
			n->exit = Asin_exit;
			n->reshape = Asin_reshape;
			n->op = Asin_float64;
			break;
		default:
			break;
		}
	}
}
#endif
