#include "uonnx.h"

#ifdef UONNX_OPS_ACOSH
static int Acosh_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Acosh_exit(Node * n)
{
	return 1;
}

static int Acosh_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Acosh_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;
	size_t i, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(acoshf(v));
	}
}

static void Acosh_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	size_t i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = acoshf(px[i]);
}

static void Acosh_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	size_t i, l;

	for(i = 0, l = y->ndata; i < l; i++)
		py[i] = acosh(px[i]);
}

void resolver_default_op_Acosh(Node * n)
{
	if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Acosh_init;
			n->exit = Acosh_exit;
			n->reshape = Acosh_reshape;
			n->op = Acosh_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Acosh_init;
			n->exit = Acosh_exit;
			n->reshape = Acosh_reshape;
			n->op = Acosh_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Acosh_init;
			n->exit = Acosh_exit;
			n->reshape = Acosh_reshape;
			n->op = Acosh_float64;
			break;
		default:
			break;
		}
	}
}
#endif