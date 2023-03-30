#include <uonnx.h>

static int Greater_init(struct onnx_node_t * n)
{
	if((n->ninput == 2) && (n->noutput == 1))
		return 1;
	return 0;
}

static int Greater_exit(struct onnx_node_t * n)
{
	return 1;
}

static int Greater_reshape(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];

	return onnx_tensor_reshape_multi_broadcast(y, a, b, ONNX_TENSOR_TYPE_BOOL);
}

static void Greater_int8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	int8_t * pa;
	int8_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_int16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	int16_t * pa;
	int16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_int32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	int32_t * pa;
	int32_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_int64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	int64_t * pa;
	int64_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_uint8(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * pa;
	uint8_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_uint16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_uint32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint32_t * pa;
	uint32_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_uint64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint64_t * pa;
	uint64_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_bfloat16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (bfloat16_to_float32(*pa) > bfloat16_to_float32(*pb)) ? 1 : 0;
	}
}

static void Greater_float16(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	uint16_t * pa;
	uint16_t * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (float16_to_float32(*pa) > float16_to_float32(*pb)) ? 1 : 0;
	}
}

static void Greater_float32(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	float * pa;
	float * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

static void Greater_float64(struct onnx_node_t * n)
{
	struct onnx_tensor_t * y = n->outputs[0];
	struct onnx_tensor_t * a = n->inputs[0];
	struct onnx_tensor_t * b = n->inputs[1];
	uint8_t * py = (uint8_t *)y->datas;
	double * pa;
	double * pb;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		pa = onnx_tensor_broadcast_map_address(a, y, i);
		pb = onnx_tensor_broadcast_map_address(b, y, i);
		py[i] = (*pa > *pb) ? 1 : 0;
	}
}

void resolver_default_op_Greater(struct onnx_node_t * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_uint64;
			break;
		case ONNX_TENSOR_TYPE_BFLOAT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_bfloat16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 9)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_INT8:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_int8;
			break;
		case ONNX_TENSOR_TYPE_INT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_int16;
			break;
		case ONNX_TENSOR_TYPE_INT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_int32;
			break;
		case ONNX_TENSOR_TYPE_INT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_int64;
			break;
		case ONNX_TENSOR_TYPE_UINT8:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_uint8;
			break;
		case ONNX_TENSOR_TYPE_UINT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_uint16;
			break;
		case ONNX_TENSOR_TYPE_UINT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_uint32;
			break;
		case ONNX_TENSOR_TYPE_UINT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_uint64;
			break;
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 7)
	{
		switch(n->inputs[0]->type)
		{
		case ONNX_TENSOR_TYPE_FLOAT16:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_float16;
			break;
		case ONNX_TENSOR_TYPE_FLOAT32:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_float32;
			break;
		case ONNX_TENSOR_TYPE_FLOAT64:
			n->init = Greater_init;
			n->exit = Greater_exit;
			n->reshape = Greater_reshape;
			n->operator = Greater_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 1)
	{
	}
}
