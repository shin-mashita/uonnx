/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_POW

static int Pow_init(Node * n)
{
	if((n->ninputs == 2) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Pow_exit(Node * n)
{
	return 1;
}

static int Pow_reshape(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];

	return onnx_tensor_reshape_multi_broadcast(y, a, b, a->type);
}

static double tensor_get_value(void * p, TensorType type)
{
	double v;

	switch(type)
	{
	case TENSOR_TYPE_BOOL:
		v = *((uint8_t *)p);
		break;
	case TENSOR_TYPE_INT8:
		v = *((int8_t *)p);
		break;
	case TENSOR_TYPE_INT16:
		v = *((int16_t *)p);
		break;
	case TENSOR_TYPE_INT32:
		v = *((int32_t *)p);
		break;
	case TENSOR_TYPE_INT64:
		v = *((int64_t *)p);
		break;
	case TENSOR_TYPE_UINT8:
		v = *((uint8_t *)p);
		break;
	case TENSOR_TYPE_UINT16:
		v = *((uint16_t *)p);
		break;
	case TENSOR_TYPE_UINT32:
		v = *((uint32_t *)p);
		break;
	case TENSOR_TYPE_UINT64:
		v = *((uint64_t *)p);
		break;
	case TENSOR_TYPE_BFLOAT16:
		v = bfloat16_to_float32(*((uint16_t *)p));
		break;
	case TENSOR_TYPE_FLOAT16:
		v = float16_to_float32(*((uint16_t *)p));
		break;
	case TENSOR_TYPE_FLOAT32:
		v = *((float *)p);
		break;
	case TENSOR_TYPE_FLOAT64:
		v = *((double *)p);
		break;
	default:
		v = 0;
		break;
	}
	return v;
}

static void Pow_int32(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	int32_t * py = (int32_t *)y->datas;
	int32_t * pa;
	void * pb;
	double v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		v = tensor_get_value(pb, b->type);
		py[i] = pow(*pa, v);
	}
}

static void Pow_int64(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	int64_t * py = (int64_t *)y->datas;
	int64_t * pa;
	void * pb;
	double v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		v = tensor_get_value(pb, b->type);
		py[i] = pow(*pa, v);
	}
}

static void Pow_bfloat16(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa;
	void * pb;
	double v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		v = tensor_get_value(pb, b->type);
		py[i] = float32_to_bfloat16(pow(bfloat16_to_float32(*pa), v));
	}
}

static void Pow_float16(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * pa;
	void * pb;
	double v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		v = tensor_get_value(pb, b->type);
		py[i] = float32_to_float16(pow(float16_to_float32(*pa), v));
	}
}

static void Pow_float32(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	float * py = (float *)y->datas;
	float * pa;
	void * pb;
	double v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		v = tensor_get_value(pb, b->type);
		py[i] = pow(*pa, v);
	}
}

static void Pow_float64(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * a = n->inputs[0];
	Tensor * b = n->inputs[1];
	double * py = (double *)y->datas;
	double * pa;
	void * pb;
	double v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		v = tensor_get_value(pb, b->type);
		py[i] = pow(*pa, v);
	}
}

void resolver_default_op_Pow(Node * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_INT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_int32;
			break;
		case TENSOR_TYPE_INT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_int64;
			break;
		case TENSOR_TYPE_BFLOAT16:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 12)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_INT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_int32;
			break;
		case TENSOR_TYPE_INT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_int64;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Pow_init;
			n->exit = Pow_exit;
			n->reshape = Pow_reshape;
			n->op = Pow_float64;
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