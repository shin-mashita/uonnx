#include "onnx_allocator.h"

static void hmap_entry_callback(struct hmap_t * m, struct hmap_entry_t * e)
{
	if(e && e->value)
		onnx_tensor_free((struct onnx_tensor_t *)e->value);
}

static int reshape_dummy(struct onnx_node_t * n)
{
	return 1;
}

static void operator_dummy(struct onnx_node_t * n)
{
	ONNX_LOG("\033[45;37mUnsupported opset\033[0m => %s-%d (%s)\r\n", n->proto->op_type, n->opset, (strlen(n->proto->domain) > 0) ? n->proto->domain : "ai.onnx");
}


struct onnx_context_t * onnx_context_alloc(const void * buf, size_t len, struct onnx_resolver_t ** r, int rlen)
{
	struct onnx_context_t * ctx;
	int i;

	if(!buf || len <= 0)
		return NULL;

	ctx = malloc(sizeof(struct onnx_context_t));
	if(!ctx)
		return NULL;

	ctx->model = onnx__model_proto__unpack(NULL, len, buf);
	if(!ctx->model)
	{
		if(ctx)
			free(ctx);
		return NULL;
	}

	ctx->map = hmap_alloc(0, hmap_entry_callback);
	if(!ctx->map)
	{
		if(ctx->model)
			onnx__model_proto__free_unpacked(ctx->model, NULL);
		if(ctx)
			free(ctx);
		return NULL;
	}

	ctx->rlen = rlen;
	if(r && (ctx->rlen > 0))
	{
		ctx->r = malloc(sizeof(struct onnx_resolver_t *) * ctx->rlen);
		ctx->rctx = malloc(sizeof(void *) * ctx->rlen);
		if(!ctx->r || !ctx->rctx)
		{
			if(ctx->rctx)
				free(ctx->rctx);
			if(ctx->r)
				free(ctx->r);
			if(ctx->map)
				hmap_free(ctx->map);
			if(ctx->model)
				onnx__model_proto__free_unpacked(ctx->model, NULL);
			if(ctx)
				free(ctx);
			return NULL;
		}
	}
	else
	{
		ctx->r = NULL;
		ctx->rctx = NULL;
	}

	for(i = 0; i < ctx->rlen; i++)
	{
		ctx->r[i] = r[i];
		if(r[i] && r[i]->create)
			ctx->rctx[i] = r[i]->create();
	}

	ctx->g = onnx_graph_alloc(ctx, ctx->model->graph);
	if(!ctx->g)
	{
		for(i = 0; i < ctx->rlen; i++)
		{
			if(ctx->r[i] && ctx->r[i]->destroy)
				ctx->r[i]->destroy(ctx->rctx[i]);
		}
		if(ctx->rctx)
			free(ctx->rctx);
		if(ctx->r)
			free(ctx->r);
		if(ctx->map)
			hmap_free(ctx->map);
		if(ctx->model)
			onnx__model_proto__free_unpacked(ctx->model, NULL);
		if(ctx)
			free(ctx);
		return NULL;
	}

	return ctx;
}

struct onnx_context_t * onnx_context_alloc_from_file(const char * filename, struct onnx_resolver_t ** r, int rlen)
{
	struct onnx_context_t * ctx = NULL;
	FILE * fp;
	void * buf;
	size_t l, len;

	fp = fopen(filename, "rb");
	if(fp)
	{
		fseek(fp, 0L, SEEK_END);
		l = ftell(fp);
		fseek(fp, 0L, SEEK_SET);
		if(l > 0)
		{
			buf = malloc(l);
			if(buf)
			{
				for(len = 0; len < l; len += fread(buf + len, 1, l - len, fp));
				ctx = onnx_context_alloc(buf, len, r, rlen);
				free(buf);
			}
		}
		fclose(fp);
	}
	return ctx;
}

void onnx_context_free(struct onnx_context_t * ctx)
{
	int i;

	if(ctx)
	{
		if(ctx->g)
			onnx_graph_free(ctx->g);
		for(i = 0; i < ctx->rlen; i++)
		{
			if(ctx->r[i] && ctx->r[i]->destroy)
				ctx->r[i]->destroy(ctx->rctx[i]);
		}
		if(ctx->rctx)
			free(ctx->rctx);
		if(ctx->r)
			free(ctx->r);
		if(ctx->map)
			hmap_free(ctx->map);
		if(ctx->model)
			onnx__model_proto__free_unpacked(ctx->model, NULL);
		free(ctx);
	}
}

static struct onnx_tensor_t * onnx_tensor_alloc_from_value_info(Onnx__ValueInfoProto * v)
{
	struct onnx_tensor_t * t;
	enum onnx_tensor_type_t type;
	int * dims = NULL;
	int ndim;
	int i;

	if(!v || !v->name)
		return NULL;

	switch(v->type->value_case)
	{
	case ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE:
		type = (enum onnx_tensor_type_t)v->type->tensor_type->elem_type;
		ndim = v->type->tensor_type->shape->n_dim;
		if(ndim > 0)
		{
			dims = malloc(sizeof(int) * ndim);
			if(dims)
			{
				for(i = 0; i < ndim; i++)
				{
					switch(v->type->tensor_type->shape->dim[i]->value_case)
					{
					case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
						dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
						break;
					case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
						if(strcmp(v->type->tensor_type->shape->dim[i]->dim_param, "batch_size") == 0)
							dims[i] = 1;
						else
							dims[i] = 1;
						break;
					default:
						dims[i] = 1;
						break;
					}
				}
			}
		}
		t = onnx_tensor_alloc(v->name, type, dims, ndim);
		if(dims)
			free(dims);
		break;
	case ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE:
		t = NULL;
		break;
	case ONNX__TYPE_PROTO__VALUE_MAP_TYPE:
		t = NULL;
		break;
	default:
		t = NULL;
		break;
	}
	return t;
}

static void onnx_tensor_copy_from_tensor_proto(struct onnx_tensor_t * t, Onnx__TensorProto * o)
{
	size_t n, i;
	int sz;

	if(t && o)
	{
		if(t->type == o->data_type)
		{\
			sz = onnx_tensor_type_sizeof(t->type);
			if(sz > 0)
			{
				if((o->raw_data.len > 0) && o->raw_data.data)
				{
					switch(o->data_type)
					{
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
						{
							float * p = (float *)t->datas;
							uint32_t * q = (uint32_t *)o->raw_data.data;
							union { uint32_t u; float f; } v;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
								{
									v.u = le32_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
						{
							uint8_t * p = (uint8_t *)t->datas;
							uint8_t * q = (uint8_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len);
								memcpy(p, q, n);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
						{
							int8_t * p = (int8_t *)t->datas;
							int8_t * q = (int8_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len);
								memcpy(p, q, n);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
						{
							uint16_t * p = (uint16_t *)t->datas;
							uint16_t * q = (uint16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
						{
							int16_t * p = (int16_t *)t->datas;
							int16_t * q = (int16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
						{
							int32_t * p = (int32_t *)t->datas;
							int32_t * q = (int32_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le32_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
						{
							int64_t * p = (int64_t *)t->datas;
							int64_t * q = (int64_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le64_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
						{
							uint8_t * p = (uint8_t *)t->datas;
							uint8_t * q = (uint8_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len);
								memcpy(p, q, n);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
						{
							uint16_t * p = (uint16_t *)t->datas;
							uint16_t * q = (uint16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
						{
							double * p = (double *)t->datas;
							uint64_t * q = (uint64_t *)o->raw_data.data;
							union { uint64_t u; double f; } v;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
								{
									v.u = le64_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
						{
							uint32_t * p = (uint32_t *)t->datas;
							uint32_t * q = (uint32_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le32_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
						{
							uint64_t * p = (uint64_t *)t->datas;
							uint64_t * q = (uint64_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le64_to_cpu(q[i]);
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
						{
							float * p = (float *)t->datas;
							uint32_t * q = (uint32_t *)o->raw_data.data;
							union { uint32_t u; float f; } v;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz) * 2;
								for(i = 0; i < n; i++)
								{
									v.u = le32_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
						{
							double * p = (double *)t->datas;
							uint64_t * q = (uint64_t *)o->raw_data.data;
							union { uint64_t u; double f; } v;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz) * 2;
								for(i = 0; i < n; i++)
								{
									v.u = le64_to_cpu(q[i]);
									p[i] = v.f;
								}
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
						{
							uint16_t * p = (uint16_t *)t->datas;
							uint16_t * q = (uint16_t *)o->raw_data.data;
							if(t->ndata > 0)
							{
								n = min(t->ndata, (size_t)o->raw_data.len / sz);
								for(i = 0; i < n; i++)
									p[i] = le16_to_cpu(q[i]);
							}
						}
						break;
					default:
						break;
					}
				}
				else
				{
					switch(o->data_type)
					{
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
						n = min(t->ndata, (size_t)o->n_float_data);
						if((n > 0) && t->datas && o->float_data)
							memcpy(t->datas, o->float_data, sizeof(float) * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT8:
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT8:
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT16:
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT16:
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT32:
					case ONNX__TENSOR_PROTO__DATA_TYPE__BOOL:
					case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16:
					case ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16:
						//TODO
						n = min(t->ndata, (size_t)o->n_int32_data);
						if((n > 0) && t->datas && o->int32_data)
							memcpy(t->datas, o->int32_data, sz * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__STRING:
						n = min(t->ndata, (size_t)o->n_string_data);
						if((n > 0) && t->datas && o->string_data)
						{
							char ** str = (char **)t->datas;
							for(i = 0; i < t->ndata; i++)
							{
								if(str[i])
								{
									free(str[i]);
									str[i] = NULL;
								}
							}
							for(i = 0; i < n; i++)
							{
								str[i] = malloc(o->string_data[i].len + 1);
								if(str[i])
								{
									str[i][o->string_data[i].len] = 0;
									memcpy(str[i], o->string_data[i].data, o->string_data[i].len);
								}
							}
						}
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__INT64:
						n = min(t->ndata, (size_t)o->n_int64_data);
						if((n > 0) && t->datas && o->int64_data)
							memcpy(t->datas, o->int64_data, sizeof(int64_t) * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE:
						n = min(t->ndata, (size_t)o->n_double_data);
						if((n > 0) && t->datas && o->double_data)
							memcpy(t->datas, o->double_data, sizeof(double) * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT32:
					case ONNX__TENSOR_PROTO__DATA_TYPE__UINT64:
						//TODO
						n = min(t->ndata, (size_t)o->n_uint64_data);
						if((n > 0) && t->datas && o->uint64_data)
							memcpy(t->datas, o->uint64_data, sz * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64:
						n = min(t->ndata, (size_t)(o->n_float_data / 2));
						if((n > 0) && t->datas && o->float_data)
							memcpy(t->datas, o->float_data, sizeof(float) * 2 * n);
						break;
					case ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128:
						n = min(t->ndata, (size_t)(o->n_double_data / 2));
						if((n > 0) && t->datas && o->double_data)
							memcpy(t->datas, o->double_data, sizeof(double) * 2 * n);
						break;
					default:
						break;
					}
				}
			}
		}
	}
}

struct onnx_graph_t * onnx_graph_alloc(struct onnx_context_t * ctx, Onnx__GraphProto * graph)
{
	struct onnx_graph_t * g;
	struct onnx_node_t * n;
	struct onnx_tensor_t * t;
	Onnx__TensorProto * o;
	Onnx__ValueInfoProto * v;
	char * p, * domain;
	char * name;
	int i, j, k, l;

	if(!graph)
		return NULL;

	// Initialize graph. Set to zero all values. 
	g = malloc(sizeof(struct onnx_graph_t));
	if(!g)
		return NULL;
	memset(g, 0, sizeof(struct onnx_graph_t));

	// Assign nlen. 
	g->nlen = graph->n_node;

	// Alloc g->nodes by size of nlen*sizeof(node).
	// MODIFY: alloc instead by max_size accdng to planner. MAX_SIZE can be passed as parameter.
	g->nodes = malloc(sizeof(struct onnx_node_t) * g->nlen);
	if(!g->nodes)
	{
		free(g);
		return NULL;
	}

	// Update map for value info inputs. 
	// SUGGESTION: remove map?
	for(i = 0; i < graph->n_input; i++)
	{
		v = graph->input[i];
		if(!onnx_tensor_search(ctx, v->name))
		{
			t = onnx_tensor_alloc_from_value_info(v);
			if(t)
			{
				for(j = 0; j < graph->n_initializer; j++)
				{
					if(strcmp(graph->initializer[j]->name, t->name) == 0)
					{
						onnx_tensor_copy_from_tensor_proto(t, graph->initializer[j]);
						break;
					}
				}
				hmap_add(ctx->map, t->name, t);
			}
		}
	}

	// Update map for value info outputs.
	for(i = 0; i < graph->n_output; i++)
	{
		v = graph->output[i];
		if(!onnx_tensor_search(ctx, v->name))
		{
			t = onnx_tensor_alloc_from_value_info(v);
			if(t)
				hmap_add(ctx->map, t->name, t);
		}
	}

	// Update map for value_info.
	for(i = 0; i < graph->n_value_info; i++)
	{
		v = graph->value_info[i];
		if(!onnx_tensor_search(ctx, v->name))
		{
			t = onnx_tensor_alloc_from_value_info(v);
			if(t)
				hmap_add(ctx->map, t->name, t);
		}
	}

	// Update map for output of each nodes.
	for(i = 0; i < graph->n_node; i++)
	{
		for(j = 0; j < graph->node[i]->n_output; j++)
		{
			name = graph->node[i]->output[j];
			if(!onnx_tensor_search(ctx, name))
			{
				t = onnx_tensor_alloc(name, ONNX_TENSOR_TYPE_UNDEFINED, NULL, 0);
				if(t)
					hmap_add(ctx->map, name, t);
			}
		}
	}

	// Update map for input of each nodes. Add each input tensors to map. 
	for(i = 0; i < graph->n_node; i++)
	{
		for(j = 0; j < graph->node[i]->n_input; j++)
		{
			name = graph->node[i]->input[j];
			if(!onnx_tensor_search(ctx, name))
			{
				for(k = 0; k < graph->n_initializer; k++)
				{
					if(strcmp(graph->initializer[k]->name, name) == 0)
					{
						o = graph->initializer[k];
						if(o)
						{
							int ndim = o->n_dims;
							int dims[ndim];
							for(l = 0; l < ndim; l++)
								dims[l] = o->dims[l];
							t = onnx_tensor_alloc(name, (enum onnx_tensor_type_t)o->data_type, dims, ndim);
							if(t)
							{
								onnx_tensor_copy_from_tensor_proto(t, o);
								hmap_add(ctx->map, name, t);
							}
							break;
						}
					}
				}
				if(!onnx_tensor_search(ctx, name))
				{
					if(g->nodes)
						free(g->nodes);
					free(g);
					return NULL;
				}
			}
		}
	}

	// To update here... 
	for(i = 0; i < g->nlen; i++)
	{
		n = &g->nodes[i];
		memset(n, 0, sizeof(struct onnx_node_t));

		n->ctx = ctx;
		n->proto = graph->node[i];
		domain = n->proto->domain;
		if(!domain || (strlen(domain) == 0))
			domain = "ai.onnx";
		for(j = 0; j < ctx->model->n_opset_import; j++)
		{
			p = ctx->model->opset_import[j]->domain;
			if(!p || (strlen(p) == 0))
				p = "ai.onnx";
			if(strcmp(domain, p) == 0)
			{
				n->opset = ctx->model->opset_import[j]->version;
				break;
			}
		}
		if(n->proto->n_input > 0)
		{
			n->inputs = malloc(sizeof(struct onnx_tensor_t *) * n->proto->n_input);
			if(n->inputs)
			{
				n->ninput = n->proto->n_input;
				for(j = 0; j < n->ninput; j++)
					n->inputs[j] = onnx_tensor_search(ctx, n->proto->input[j]);
			}
		}
		if(n->proto->n_output > 0)
		{
			n->outputs = malloc(sizeof(struct onnx_tensor_t *) * n->proto->n_output);
			if(n->outputs)
			{
				n->noutput = n->proto->n_output;
				for(j = 0; j < n->noutput; j++)
					n->outputs[j] = onnx_tensor_search(ctx, n->proto->output[j]);
			}
		}
		for(j = 0; j < ctx->rlen; j++)
		{
			resolver_solve_operator(ctx->r[j], n);
			if(n->operator)
			{
				n->r = ctx->r[j];
				n->rctx = ctx->rctx[j];
				break;
			}
		}
		if(!n->operator)
		{
			resolver_solve_operator(&resolver_default, n);
			if(n->operator)
			{
				n->r = &resolver_default;
				n->rctx = NULL;
			}
		}
		if(!n->reshape)
			n->reshape = reshape_dummy;
		if(!n->operator)
			n->operator = operator_dummy;
		if(n->init)
		{
			if(n->init(n) <= 0)
			{
				if(g->nodes)
				{
					for(j = 0; j <= i; j++)
					{
						n = &g->nodes[j];
						if(n->exit)
							n->exit(n);
						if(n->inputs)
							free(n->inputs);
						if(n->outputs)
							free(n->outputs);
					}
					free(g->nodes);
				}
				free(g);
				return NULL;
			}
		}
		if(n->reshape)
			n->reshape(n);
	}

	return g;
}

void onnx_graph_free(struct onnx_graph_t * g)
{
	struct onnx_node_t * n;
	int i;

	if(g)
	{
		if(g->nodes)
		{
			for(i = 0; i < g->nlen; i++)
			{
				n = &g->nodes[i];
				if(n->exit)
					n->exit(n);
				if(n->inputs)
					free(n->inputs);
				if(n->outputs)
					free(n->outputs);
			}
			free(g->nodes);
		}
		free(g);
	}
}

struct onnx_tensor_t * onnx_tensor_alloc(const char * name, enum onnx_tensor_type_t type, int * dims, int ndim)
{
	struct onnx_tensor_t * t;

	if(!name)
		return NULL;

	t = malloc(sizeof(struct onnx_tensor_t));
	if(!t)
		return NULL;
	memset(t, 0, sizeof(struct onnx_tensor_t));

	t->name = strdup(name);
	onnx_tensor_reinit(t, type, dims, ndim);
	return t;
}

struct onnx_tensor_t * onnx_tensor_alloc_from_file(const char * filename)
{
	struct onnx_tensor_t * t = NULL;
	Onnx__TensorProto * pb;
	FILE * fp;
	void * buf;
	size_t l, len;
	int * dims = NULL;
	int ndim = 0;
	int i;

	fp = fopen(filename, "rb");
	if(fp)
	{
		fseek(fp, 0L, SEEK_END);
		l = ftell(fp);
		fseek(fp, 0L, SEEK_SET);
		if(l > 0)
		{
			buf = malloc(l);
			if(buf)
			{
				for(len = 0; len < l; len += fread(buf + len, 1, l - len, fp));
				pb = onnx__tensor_proto__unpack(NULL, len, buf);
				free(buf);
				if(pb)
				{
					if(pb->n_dims > 0)
					{
						dims = malloc(sizeof(int) * pb->n_dims);
						if(dims)
						{
							for(i = 0; i < pb->n_dims; i++)
								dims[i] = pb->dims[i];
							ndim = pb->n_dims;
						}
					}
					t = onnx_tensor_alloc(pb->name, (enum onnx_tensor_type_t)pb->data_type, dims, ndim);
					if((ndim > 0) && dims)
						free(dims);
					onnx_tensor_copy_from_tensor_proto(t, pb);
					onnx__tensor_proto__free_unpacked(pb, NULL);
				}
			}
		}
		fclose(fp);
	}
	return t;
}

void onnx_tensor_free(struct onnx_tensor_t * t)
{
	char ** str;

	if(t)
	{
		if(t->name)
			free(t->name);
		if(t->ndim > 0)
		{
			if(t->strides)
				free(t->strides);
			if(t->dims)
				free(t->dims);
		}
		if((t->ndata > 0) && t->datas)
		{
			if(t->type == ONNX_TENSOR_TYPE_STRING)
			{
				str = (char **)t->datas;
				for(size_t idx = 0; idx < t->ndata; idx++)
				{
					if(str[idx])
						free(str[idx]);
				}
			}
			free(t->datas);
		}
		free(t);
	}
}