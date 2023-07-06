/**
 * Operator reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx.h"

#ifdef UONNX_OPS_GLOBALLPPOOL

struct operator_pdata_t {
	float p;
};

static int GlobalLpPool_init(Node * n)
{
	struct operator_pdata_t * pdat;

	if((n->ninputs == 1) && (n->noutputs == 1))
	{
		pdat = malloc(sizeof(struct operator_pdata_t));
		if(pdat)
		{
			if(n->opset >= 2)
				pdat->p = onnx_attribute_read_int(n, "p", 2);
			else
				pdat->p = onnx_attribute_read_float(n, "p", 2.0);
			n->priv = pdat;
			return 1;
		}
	}
	return 0;
}

static int GlobalLpPool_exit(Node * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;

	if(pdat)
		free(pdat);
	return 1;
}

static int GlobalLpPool_reshape(Node * n)
{
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	int ndim = x->ndim;
	int dims[ndim];
	int i;

	for(i = 0; i < ndim; i++)
	{
		if(i < 2)
			dims[i] = x->dims[i];
		else
			dims[i] = 1;
	}
	return onnx_tensor_reshape(y, dims, ndim, x->type);
}

static void GlobalLpPool_float16(Node * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	uint16_t * px = (uint16_t *)x->datas;
	uint16_t * py = (uint16_t *)y->datas;
	float v;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < C; ++j)
		{
			o = i * C + j;
			for(k = 0, v = float16_to_float32(0); k < m; ++k)
				v += powf(fabsf(float16_to_float32(px[o * m + k])), pdat->p);
			py[o] = float32_to_float16(powf(v, 1.0 / pdat->p));
		}
	}
}

static void GlobalLpPool_float32(Node * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	float * px = (float *)x->datas;
	float * py = (float *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < C; ++j)
		{
			o = i * C + j;
			for(k = 0, py[o] = 0; k < m; ++k)
				py[o] += powf(fabsf(px[o * m + k]), pdat->p);
			py[o] = powf(py[o], 1.0 / pdat->p);
		}
	}
}

static void GlobalLpPool_float64(Node * n)
{
	struct operator_pdata_t * pdat = (struct operator_pdata_t *)n->priv;
	Tensor * x = n->inputs[0];
	Tensor * y = n->outputs[0];
	double * px = (double *)x->datas;
	double * py = (double *)y->datas;
	int N = y->dims[0];
	int C = y->dims[1];
	int m = x->strides[1];
	int i, j, k, o;

	for(i = 0; i < N; ++i)
	{
		for(j = 0; j < C; ++j)
		{
			o = i * C + j;
			for(k = 0, py[o] = 0; k < m; ++k)
				py[o] += pow(fabs(px[o * m + k]), pdat->p);
			py[o] = pow(py[o], 1.0 / pdat->p);
		}
	}
}

void resolver_default_op_GlobalLpPool(Node * n)
{
	if(n->opset >= 2)
	{
		switch(n->inputs[0]->type)
		{
		case TENSOR_TYPE_FLOAT16:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->op = GlobalLpPool_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->op = GlobalLpPool_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->op = GlobalLpPool_float64;
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
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->op = GlobalLpPool_float16;
			break;
		case TENSOR_TYPE_FLOAT32:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->op = GlobalLpPool_float32;
			break;
		case TENSOR_TYPE_FLOAT64:
			n->init = GlobalLpPool_init;
			n->exit = GlobalLpPool_exit;
			n->reshape = GlobalLpPool_reshape;
			n->op = GlobalLpPool_float64;
			break;
		default:
			break;
		}
	}
}

#endif