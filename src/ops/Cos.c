#include "uonnx.h"

#ifdef UONNX_OPS_COS

static int Cos_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Cos_exit(Node * n)
{
	return 1;
}

static int Cos_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Cos_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(cosf(v));
	}
}

static void Cos_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = cosf(px[i]);
}

static void Cos_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = cos(px[i]);
}

void resolver_default_op_Cos(Node * n)
{
	if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Cos_init;
			n->exit = Cos_exit;
			n->reshape = Cos_reshape;
			n->op = Cos_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Cos_init;
			n->exit = Cos_exit;
			n->reshape = Cos_reshape;
			n->op = Cos_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Cos_init;
			n->exit = Cos_exit;
			n->reshape = Cos_reshape;
			n->op = Cos_float64;
			break;
		default:
			break;
		}
	}
}

#endif