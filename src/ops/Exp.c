#include "uonnx.h"

#ifdef UONNX_OPS_EXP

static int Exp_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Exp_exit(Node * n)
{
	return 1;
}

static int Exp_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Exp_bfloat16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		py[i] = float32_to_bfloat16(expf(v));
	}
}

static void Exp_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(expf(v));
	}
}

static void Exp_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = expf(px[i]);
}

static void Exp_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = exp(px[i]);
}

void resolver_default_op_Exp(Node * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BFLOAT16:
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_float64;
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
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_float64;
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
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Exp_init;
			n->exit = Exp_exit;
			n->reshape = Exp_reshape;
			n->op = Exp_float64;
			break;
		default:
			break;
		}
	}
}

#endif