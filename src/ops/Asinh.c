#include "uonnx.h"

#ifdef UONNX_OPS_ASINH
static int Asinh_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Asinh_exit(Node * n)
{
	return 1;
}

static int Asinh_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Asinh_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(asinhf(v));
	}
}

static void Asinh_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = asinhf(px[i]);
}

static void Asinh_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = asinh(px[i]);
}

void resolver_default_op_Asinh(Node * n)
{
	if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Asinh_init;
			n->exit = Asinh_exit;
			n->reshape = Asinh_reshape;
			n->op = Asinh_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Asinh_init;
			n->exit = Asinh_exit;
			n->reshape = Asinh_reshape;
			n->op = Asinh_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Asinh_init;
			n->exit = Asinh_exit;
			n->reshape = Asinh_reshape;
			n->op = Asinh_float64;
			break;
		default:
			break;
		}
	}
}
#endif