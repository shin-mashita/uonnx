/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

static int Mean_init(Node * n)
{
	if((n->ninputs >= 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Mean_exit(Node * n)
{
	return 1;
}

static int Mean_reshape(Node * n)
{
	Tensor * y = n->outputs[0];
	int i;

	if(!onnx_tensor_reshape_identity(y, n->inputs[0], n->inputs[0]->type))
		return 0;
	for(i = 1; i < n->ninputs; i++)
	{
		if(!onnx_tensor_reshape_multi_broadcast(y, y, n->inputs[i], y->type))
			return 0;
	}
	return 1;
}

static void Mean_bfloat16(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float sum;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, sum = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			sum += bfloat16_to_float32(*px);
		}
		py[i] = float32_to_bfloat16(sum / n->ninputs);
	}
}

static void Mean_float16(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float sum;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, sum = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			sum += float16_to_float32(*px);
		}
		py[i] = float32_to_float16(sum / n->ninputs);
	}
}

static void Mean_float32(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	float * py = (float *)y->datas;
	float * px;
	float sum;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, sum = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			sum += *px;
		}
		py[i] = sum / n->ninputs;
	}
}

static void Mean_float64(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	double * py = (double *)y->datas;
	double * px;
	double sum;
	size_t i, j, l;

	for(i = 0, l = y->ndata; i < l; i++)
	{
		for(j = 0, sum = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			sum += *px;
		}
		py[i] = sum / n->ninputs;
	}
}

void resolver_default_op_Mean(Node * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BFLOAT16:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 8)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float64;
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
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float64;
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
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Mean_init;
			n->exit = Mean_exit;
			n->reshape = Mean_reshape;
			n->op = Mean_float64;
			break;
		default:
			break;
		}
	}
}
