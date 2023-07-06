/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_ABS
static int Abs_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Abs_exit(Node * n)
{
	return 1;
}

static int Abs_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Abs_int8(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int8_t * px = (int8_t *)x->datas;
	int8_t * py = (int8_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_int16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int16_t * px = (int16_t *)x->datas;
	int16_t * py = (int16_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_int32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int32_t * px = (int32_t *)x->datas;
	int32_t * py = (int32_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_int64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int64_t * px = (int64_t *)x->datas;
	int64_t * py = (int64_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_uint8(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint8_t * px = (uint8_t *)x->datas;
	uint8_t * py = (uint8_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_uint16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_uint32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint32_t * px = (uint32_t *)x->datas;
	uint32_t * py = (uint32_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_uint64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint64_t * px = (uint64_t *)x->datas;
	uint64_t * py = (uint64_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? -px[i] : px[i];
}

static void Abs_bfloat16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		py[i] = float32_to_bfloat16(fabsf(v));
	}
}

static void Abs_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16(fabsf(v));
	}
}

static void Abs_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = fabsf(px[i]);
}

static void Abs_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = fabs(px[i]);
}

void resolver_default_op_Abs(Node * n)
{
	if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_INT8:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_int8;
			break;
		case TENSOR_TYPE_INT16:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_int16;
			break;
		case TENSOR_TYPE_INT32:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_int32;
			break;
		case TENSOR_TYPE_INT64:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_int64;
			break;
		case TENSOR_TYPE_UINT8:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_uint8;
			break;
		case TENSOR_TYPE_UINT16:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_uint16;
			break;
		case TENSOR_TYPE_UINT32:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_uint32;
			break;
		case TENSOR_TYPE_UINT64:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_uint64;
			break;
		case TENSOR_TYPE_BFLOAT16:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_INT8:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_int8;
			break;
		case TENSOR_TYPE_INT16:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_int16;
			break;
		case TENSOR_TYPE_INT32:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_int32;
			break;
		case TENSOR_TYPE_INT64:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_int64;
			break;
		case TENSOR_TYPE_UINT8:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_uint8;
			break;
		case TENSOR_TYPE_UINT16:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_uint16;
			break;
		case TENSOR_TYPE_UINT32:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_uint32;
			break;
		case TENSOR_TYPE_UINT64:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_uint64;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_float64;
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
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Abs_init;
			n->exit = Abs_exit;
			n->reshape = Abs_reshape;
			n->op = Abs_float64;
			break;
		default:
			break;
		}
	}
}

#endif