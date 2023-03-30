#include "onnx_logger.h"

void onnx_model_dump(Onnx__ModelProto * model)
{
    int i;
    if(model)
    {
        ONNX_LOG("IR Version: v%ld\r\n", model->ir_version);
        ONNX_LOG("Producer: %s %s\r\n", model->producer_name, model->producer_version);
        ONNX_LOG("Domain: %s\r\n", model->domain);
        ONNX_LOG("Imports:\r\n");
        for(i = 0; i < model->n_opset_import; i++)
            ONNX_LOG("\t%s v%ld\r\n", (strlen(model->opset_import[i]->domain) > 0) ? model->opset_import[i]->domain : "ai.onnx", model->opset_import[i]->version);
    }
    else
    {
        ONNX_LOG("Blank model.");
    }
}

void onnx_tensor_dump(struct onnx_tensor_t * t, int detail)
{
	int * sizes, * levels;
	char * lbuf, * rbuf;
	char * lp, * rp;
	void * p;
	int i, j, k;

	if(t)
	{
		ONNX_LOG("%s: %s", t->name, onnx_tensor_type_tostring(t->type));
		if(t->ndim > 0)
		{
			ONNX_LOG("[");
			for(i = 0; i < t->ndim; i++)
			{
				ONNX_LOG("%d", t->dims[i]);
				if(i != t->ndim - 1)
					ONNX_LOG(" x ");
			}
			ONNX_LOG("]");
			if(detail)
			{
				ONNX_LOG(" = \r\n");
				for(i = 0; i < t->ndim; i++)
				{
					if(t->dims[i] <= 0)
						return;
				}
				sizes = malloc(sizeof(int) * t->ndim);
				levels = malloc(sizeof(int) * t->ndim);
				sizes[t->ndim - 1] = t->dims[t->ndim - 1];
				levels[t->ndim - 1] = 0;
				lbuf = malloc(sizeof(char) * (t->ndim + 1));
				rbuf = malloc(sizeof(char) * (t->ndim + 1));
				lp = lbuf;
				rp = rbuf;
				for(i = t->ndim - 2; i >= 0; i--)
				{
					sizes[i] = t->dims[i] * sizes[i + 1];
					levels[i] = 0;
				}
				for(size_t idx = 0; idx < t->ndata; idx++)
				{
					for(j = 0; j < t->ndim; j++)
					{
						if((idx % sizes[j]) == 0)
							levels[j]++;
						if(levels[j] == 1)
						{
							*lp++ = '[';
							levels[j]++;
						}
						if(levels[j] == 3)
						{
							*rp++ = ']';
							if((j != 0) && (levels[j] > levels[j - 1]))
							{
								*lp++ = '[';
								levels[j] = 2;
							}
							else
							{
								levels[j] = 0;
							}
						}
					}
					*lp = *rp = '\0';
					ONNX_LOG("%s", rbuf);
					if(*rbuf != '\0')
					{
						ONNX_LOG("\r\n");
						for(k = t->ndim - strlen(rbuf); k > 0; k--)
							ONNX_LOG(" ");
					}
					ONNX_LOG("%s", lbuf);
					if(*lbuf == '\0')
						ONNX_LOG(" ");
					p = (void *)(t->datas + onnx_tensor_type_sizeof(t->type) * idx);
					switch(t->type)
					{
					case ONNX_TENSOR_TYPE_BOOL:
						ONNX_LOG("%s,", *((uint8_t *)p) ? "true" : "false");
						break;
					case ONNX_TENSOR_TYPE_INT8:
						ONNX_LOG("%d,", *((int8_t *)p));
						break;
					case ONNX_TENSOR_TYPE_INT16:
						ONNX_LOG("%d,", *((int16_t *)p));
						break;
					case ONNX_TENSOR_TYPE_INT32:
						ONNX_LOG("%d,", *((int32_t *)p));
						break;
					case ONNX_TENSOR_TYPE_INT64:
						ONNX_LOG("%ld,", *((int64_t *)p));
						break;
					case ONNX_TENSOR_TYPE_UINT8:
						ONNX_LOG("%u,", *((uint8_t *)p));
						break;
					case ONNX_TENSOR_TYPE_UINT16:
						ONNX_LOG("%u,", *((uint16_t *)p));
						break;
					case ONNX_TENSOR_TYPE_UINT32:
						ONNX_LOG("%u,", *((uint32_t *)p));
						break;
					case ONNX_TENSOR_TYPE_UINT64:
						ONNX_LOG("%lu,", *((uint64_t *)p));
						break;
					case ONNX_TENSOR_TYPE_BFLOAT16:
						ONNX_LOG("%g,", bfloat16_to_float32(*((uint16_t *)p)));
						break;
					case ONNX_TENSOR_TYPE_FLOAT16:
						ONNX_LOG("%g,", float16_to_float32(*((uint16_t *)p)));
						break;
					case ONNX_TENSOR_TYPE_FLOAT32:
						ONNX_LOG("%g,", *((float *)p));
						break;
					case ONNX_TENSOR_TYPE_FLOAT64:
						ONNX_LOG("%g,", *((double *)p));
						break;
					case ONNX_TENSOR_TYPE_COMPLEX64:
						ONNX_LOG("%g + %gi,", *((float *)p), *((float *)(p + sizeof(float))));
						break;
					case ONNX_TENSOR_TYPE_COMPLEX128:
						ONNX_LOG("%g + %gi,", *((double *)p), *((double *)(p + sizeof(double))));
						break;
					case ONNX_TENSOR_TYPE_STRING:
						ONNX_LOG("%s,", (char *)(((char **)p)[0]));
						break;
					default:
						ONNX_LOG("?,");
						break;
					}
					lp = lbuf;
					rp = rbuf;
				}
				for(j = 0; j < t->ndim; j++)
					ONNX_LOG("]");
				free(sizes);
				free(levels);
				free(lbuf);
				free(rbuf);
				ONNX_LOG("\r\n");
			}
			else
			{
				ONNX_LOG(" = ");
				ONNX_LOG("[...]");
				ONNX_LOG("\r\n");
			}
		}
		else if(t->ndata == 1)
		{
			ONNX_LOG(" = ");
			p = (void *)(t->datas);
			switch(t->type)
			{
			case ONNX_TENSOR_TYPE_BOOL:
				ONNX_LOG("%s", *((uint8_t *)p) ? "true" : "false");
				break;
			case ONNX_TENSOR_TYPE_INT8:
				ONNX_LOG("%d", *((int8_t *)p));
				break;
			case ONNX_TENSOR_TYPE_INT16:
				ONNX_LOG("%d", *((int16_t *)p));
				break;
			case ONNX_TENSOR_TYPE_INT32:
				ONNX_LOG("%d", *((int32_t *)p));
				break;
			case ONNX_TENSOR_TYPE_INT64:
				ONNX_LOG("%ld", *((int64_t *)p));
				break;
			case ONNX_TENSOR_TYPE_UINT8:
				ONNX_LOG("%u", *((uint8_t *)p));
				break;
			case ONNX_TENSOR_TYPE_UINT16:
				ONNX_LOG("%u", *((uint16_t *)p));
				break;
			case ONNX_TENSOR_TYPE_UINT32:
				ONNX_LOG("%u", *((uint32_t *)p));
				break;
			case ONNX_TENSOR_TYPE_UINT64:
				ONNX_LOG("%lu", *((uint64_t *)p));
				break;
			case ONNX_TENSOR_TYPE_BFLOAT16:
				ONNX_LOG("%g", bfloat16_to_float32(*((uint16_t *)p)));
				break;
			case ONNX_TENSOR_TYPE_FLOAT16:
				ONNX_LOG("%g", float16_to_float32(*((uint16_t *)p)));
				break;
			case ONNX_TENSOR_TYPE_FLOAT32:
				ONNX_LOG("%g", *((float *)p));
				break;
			case ONNX_TENSOR_TYPE_FLOAT64:
				ONNX_LOG("%g", *((double *)p));
				break;
			case ONNX_TENSOR_TYPE_COMPLEX64:
				ONNX_LOG("%g + %gi", *((float *)p), *((float *)(p + sizeof(float))));
				break;
			case ONNX_TENSOR_TYPE_COMPLEX128:
				ONNX_LOG("%g + %gi", *((double *)p), *((double *)(p + sizeof(double))));
				break;
			case ONNX_TENSOR_TYPE_STRING:
				ONNX_LOG("%s", (char *)(((char **)p)[0]));
				break;
			default:
				ONNX_LOG("?");
				break;
			}
			ONNX_LOG("\r\n");
		}
		else
		{
			ONNX_LOG(" = ");
			ONNX_LOG("null");
			ONNX_LOG("\r\n");
		}
	}
}

void onnx_node_dump(struct onnx_node_t * n, int detail)
{
	int i;

	if(n)
	{
		ONNX_LOG("%s: %s-%d (%s)\r\n", n->proto->name, n->proto->op_type, n->opset, (strlen(n->proto->domain) > 0) ? n->proto->domain : "ai.onnx");
		if(n->ninput > 0)
		{
			ONNX_LOG("\tInputs:\r\n");
			for(i = 0; i < n->ninput; i++)
			{
				ONNX_LOG("\t\t");
				onnx_tensor_dump(n->inputs[i], detail);
			}
		}
		if(n->noutput > 0)
		{
			ONNX_LOG("\tOutputs:\r\n");
			for(i = 0; i < n->noutput; i++)
			{
				ONNX_LOG("\t\t");
				onnx_tensor_dump(n->outputs[i], detail);
			}
		}
	}
}

void onnx_graph_dump(struct onnx_graph_t * g, int detail)
{
	int i;

	if(g)
	{
		for(i = 0; i < g->nlen; i++)
			onnx_node_dump(&g->nodes[i], detail);
	}
}
void onnx_context_dump(struct onnx_context_t * ctx, int detail)
{
	int i;

	if(ctx)
	{
		if(ctx->model)
		{
			ONNX_LOG("IR Version: v%ld\r\n", ctx->model->ir_version);
			ONNX_LOG("Producer: %s %s\r\n", ctx->model->producer_name, ctx->model->producer_version);
			ONNX_LOG("Domain: %s\r\n", ctx->model->domain);
			ONNX_LOG("Imports:\r\n");
			for(i = 0; i < ctx->model->n_opset_import; i++)
				ONNX_LOG("\t%s v%ld\r\n", (strlen(ctx->model->opset_import[i]->domain) > 0) ? ctx->model->opset_import[i]->domain : "ai.onnx", ctx->model->opset_import[i]->version);
		}
		if(ctx->g)
			onnx_graph_dump(ctx->g, detail);
	}
}