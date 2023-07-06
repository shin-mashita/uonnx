/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_AND
static int And_7_init(Node * n)
{
	if((n->ninputs == 2) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int And_7_exit(Node * n)
{
	return 1;
}

static int And_7_reshape(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];

	return onnx_tensor_reshape_multi_broadcast(y, a, b, TENSOR_TYPE_BOOL);
}

static void And_7_bool(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * pa;
	uint8_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa && *pb) ? 1 : 0;
	}
}

void resolver_default_op_And(Node * n)
{
	if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BOOL:
			n->init = And_7_init;
			n->exit = And_7_exit;
			n->reshape = And_7_reshape;
			n->op = And_7_bool;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
#endif