#include "uonnx.h"

#ifdef UONNX_OPS_MAX

static int Max_init(Node * n)
{
	if((n->ninputs >= 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Max_exit(Node * n)
{
	return 1;
}

static int Max_reshape(Node * n)
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

static void Max_int8(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	int8_t * py = (int8_t *)y->datas;
	int8_t * px;
	int8_t maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = INT8_MIN;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_int16(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	int16_t * py = (int16_t *)y->datas;
	int16_t * px;
	int16_t maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = INT16_MIN;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_int32(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	int32_t * py = (int32_t *)y->datas;
	int32_t * px;
	int32_t maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = INT32_MIN;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_int64(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	int64_t * py = (int64_t *)y->datas;
	int64_t * px;
	int64_t maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = INT64_MIN;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_uint8(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	uint8_t * py = (uint8_t *)y->datas;
	uint8_t * px;
	uint8_t maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = 0;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_uint16(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	uint16_t maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = 0;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_uint32(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	uint32_t * py = (uint32_t *)y->datas;
	uint32_t * px;
	uint32_t maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = 0;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_uint64(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	uint64_t * py = (uint64_t *)y->datas;
	uint64_t * px;
	uint64_t maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = 0;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_bfloat16(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float v;
	float maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = FLT_MIN;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			v = bfloat16_to_float32(*px);
			if(v > maxv)
				maxv = v;
		}
		py[i] = float32_to_bfloat16(maxv);
	}
}

static void Max_float16(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	uint16_t * py = (uint16_t *)y->datas;
	uint16_t * px;
	float v;
	float maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = FLT_MIN;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			v = float16_to_float32(*px);
			if(v > maxv)
				maxv = v;
		}
		py[i] = float32_to_float16(maxv);
	}
}

static void Max_float32(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	float * py = (float *)y->datas;
	float * px;
	float maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = FLT_MIN;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

static void Max_float64(Node * n)
{
	Tensor * y = n->outputs[0];
	Tensor * x;
	double * py = (double *)y->datas;
	double * px;
	double maxv;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		maxv = DBL_MIN;
		for(int j = 0; j < n->ninputs; j++)
		{
			x = n->inputs[j];
			px = onnx_tensor_broadcast_map_address(x, y, i);
			if(*px > maxv)
				maxv = *px;
		}
		py[i] = maxv;
	}
}

void resolver_default_op_Max(Node * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_INT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_int8;
			break;
		case TENSOR_TYPE_INT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_int16;
			break;
		case TENSOR_TYPE_INT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_int32;
			break;
		case TENSOR_TYPE_INT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_int64;
			break;
		case TENSOR_TYPE_UINT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_uint8;
			break;
		case TENSOR_TYPE_UINT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_uint16;
			break;
		case TENSOR_TYPE_UINT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_uint32;
			break;
		case TENSOR_TYPE_UINT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_uint64;
			break;
		case TENSOR_TYPE_BFLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 12)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_INT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_int8;
			break;
		case TENSOR_TYPE_INT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_int16;
			break;
		case TENSOR_TYPE_INT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_int32;
			break;
		case TENSOR_TYPE_INT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_int64;
			break;
		case TENSOR_TYPE_UINT8:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_uint8;
			break;
		case TENSOR_TYPE_UINT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_uint16;
			break;
		case TENSOR_TYPE_UINT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_uint32;
			break;
		case TENSOR_TYPE_UINT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_uint64;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float64;
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
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float64;
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
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float64;
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
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Max_init;
			n->exit = Max_exit;
			n->reshape = Max_reshape;
			n->op = Max_float64;
			break;
		default:
			break;
		}
	}
}

#endif
