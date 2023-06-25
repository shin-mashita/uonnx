#include <uonnx.h>

static int Relu_init(Node * n)
{
	if((n->ninputs == 1) && (n->noutputs == 1))
		return 1;
	return 0;
}

static int Relu_exit(Node * n)
{
	return 1;
}

static int Relu_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Relu_int8(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int8_t * px = (int8_t *)x->datas;
	int8_t * py = (int8_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? 0 : px[i];
}

static void Relu_int16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int16_t * px = (int16_t *)x->datas;
	int16_t * py = (int16_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? 0 : px[i];
}

static void Relu_int32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int32_t * px = (int32_t *)x->datas;
	int32_t * py = (int32_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? 0 : px[i];
}

static void Relu_int64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int64_t * px = (int64_t *)x->datas;
	int64_t * py = (int64_t *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? 0 : px[i];
}

static void Relu_bfloat16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = bfloat16_to_float32(px[i]);
		if(v < 0)
			v = 0;
		py[i] = float32_to_bfloat16(v);
	}
}

static void Relu_float16(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		if(v < 0)
			v = 0;
		py[i] = float32_to_float16(v);
	}
}

static void Relu_float32(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? 0 : px[i];
}

static void Relu_float64(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? 0 : px[i];
}

void resolver_default_op_Relu(Node * n)
{
	if(n->opset >= 14)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_INT8:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_int8;
			break;
		case TENSOR_TYPE_INT16:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_int16;
			break;
		case TENSOR_TYPE_INT32:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_int32;
			break;
		case TENSOR_TYPE_INT64:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_int64;
			break;
		case TENSOR_TYPE_BFLOAT16:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float64;
			break;
		default:
			break;
		}
	}
	else if(n->opset >= 13)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_BFLOAT16:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_bfloat16;
			break;
		case TENSOR_TYPE_FLOAT16:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float64;
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
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float64;
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
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Relu_init;
			n->exit = Relu_exit;
			n->reshape = Relu_reshape;
			n->op = Relu_float64;
			break;
		default:
			break;
		}
	}
}
