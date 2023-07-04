#include "uonnx.h"

#ifdef UONNX_OPS_ELU

struct operator_pdata_t {
	float alpha;
};

static int Elu_init(Node * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninputs == 1) && (n->noutputs == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			pdat->alpha = onnx_attribute_read_float(n, "alpha", 1.0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int Elu_exit(Node * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int Elu_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];

	return onnx_tensor_reshape_identity(y, x, x->type);
}

static void Elu_float16(Node * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;

	for(size_t i = 0, l = y->ndata; i < l; i++)
	{
		v = float16_to_float32(px[i]);
		py[i] = float32_to_float16((px[i] < 0) ? (expf(v) - 1) * pdat->alpha : v);
	}
}

static void Elu_float32(Node * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? (expf(px[i]) - 1) * pdat->alpha : px[i];
}

static void Elu_float64(Node * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;

	for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? (exp(px[i]) - 1) * pdat->alpha : px[i];
}

void resolver_default_op_Elu(Node * n)
{
	if(n->opset >= 6)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = Elu_init;
			n->exit = Elu_exit;
			n->reshape = Elu_reshape;
			n->op = Elu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Elu_init;
			n->exit = Elu_exit;
			n->reshape = Elu_reshape;
			n->op = Elu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Elu_init;
			n->exit = Elu_exit;
			n->reshape = Elu_reshape;
			n->op = Elu_float64;
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
			n->init = Elu_init;
			n->exit = Elu_exit;
			n->reshape = Elu_reshape;
			n->op = Elu_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = Elu_init;
			n->exit = Elu_exit;
			n->reshape = Elu_reshape;
			n->op = Elu_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = Elu_init;
			n->exit = Elu_exit;
			n->reshape = Elu_reshape;
			n->op = Elu_float64;
			break;
		default:
			break;
		}
	}
}

#endif