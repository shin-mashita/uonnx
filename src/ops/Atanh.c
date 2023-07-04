#include "uonnx.h"

#ifdef UONNX_OPS_ATANH
static int Atanh_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Atanh_exit(Node * n)
{
	return 1;
}

static int Atanh_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Atanh_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(atanhf(v));
	}
}

static void Atanh_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = atanhf(px[i]);
}

static void Atanh_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = atanh(px[i]);
}

void resolver_default_op_Atanh(Node * n)
{
	if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Atanh_init;
			n->exit = Atanh_exit;
			n->reshape = Atanh_reshape;
			n->op = Atanh_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Atanh_init;
			n->exit = Atanh_exit;
			n->reshape = Atanh_reshape;
			n->op = Atanh_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Atanh_init;
			n->exit = Atanh_exit;
			n->reshape = Atanh_reshape;
			n->op = Atanh_float64;
			break;
		default:
			break;
		}
	}
}
#endif
