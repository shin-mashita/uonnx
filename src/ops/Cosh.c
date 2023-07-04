#include "uonnx.h"

#ifdef UONNX_OPS_COSH

static int Cosh_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Cosh_exit(Node * n)
{
	return 1;
}

static int Cosh_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Cosh_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(coshf(v));
	}
}

static void Cosh_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = coshf(px[i]);
}

static void Cosh_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = cosh(px[i]);
}

void resolver_default_op_Cosh(Node * n)
{
	if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Cosh_init;
			n->exit = Cosh_exit;
			n->reshape = Cosh_reshape;
			n->op = Cosh_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Cosh_init;
			n->exit = Cosh_exit;
			n->reshape = Cosh_reshape;
			n->op = Cosh_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Cosh_init;
			n->exit = Cosh_exit;
			n->reshape = Cosh_reshape;
			n->op = Cosh_float64;
			break;
		default:
			break;
		}
	}
}

#endif