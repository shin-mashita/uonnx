/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_SIGMOID

static int Sigmoid_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Sigmoid_exit(Node * n)
{
	return 1;
}

static int Sigmoid_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Sigmoid_bfloat16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		if(v >= 0)
			py[i] = float32_to_bfloat16(1.0 / (1.0 + expf(-1 * v)));
		else
			py[i] = float32_to_bfloat16(expf(v) / (1.0 + expf(v)));
	}
}

static void Sigmoid_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		if(v >= 0)
			py[i] = float32_to_float16(1.0 / (1.0 + expf(-1 * v)));
		else
			py[i] = float32_to_float16(expf(v) / (1.0 + expf(v)));
	}
}

static void Sigmoid_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] >= 0)
			py[i] = 1.0 / (1.0 + expf(-1 * px[i]));
		else
			py[i] = expf(px[i]) / (1.0 + expf(px[i]));
	}
}

static void Sigmoid_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		if(px[i] >= 0)
			py[i] = 1.0 / (1.0 + exp(-1 * px[i]));
		else
			py[i] = exp(px[i]) / (1.0 + exp(px[i]));
	}
}

void resolver_default_op_Sigmoid(Node * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BFLOAT16:
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_float64;
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
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_float64;
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
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Sigmoid_init;
			n->exit = Sigmoid_exit;
			n->reshape = Sigmoid_reshape;
			n->op = Sigmoid_float64;
			break;
		default:
			break;
		}
	}
}

#endif