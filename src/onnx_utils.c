#include "onnx_utils.h"

const char * onnx_tensor_type_tostring(enum onnx_tensor_type_t type)
{
	static const char * typestr[17] = {
		"undefined",
		"float32",
		"uint8",
		"int8",
		"uint16",
		"int16",
		"int32",
		"int64",
		"string",
		"bool",
		"float16",
		"float64",
		"uint32",
		"uint64",
		"complex64",
		"complex128",
		"bfloat16",
	};
	if((type > 0) && (type < (sizeof(typestr) / sizeof((typestr)[0]))))
		return typestr[type];
	return typestr[0];
}

int onnx_tensor_type_sizeof(enum onnx_tensor_type_t type)
{
	static const int typesz[17] = {
		0,
		sizeof(float),
		sizeof(uint8_t),
		sizeof(int8_t),
		sizeof(uint16_t),
		sizeof(int16_t),
		sizeof(int32_t),
		sizeof(int64_t),
		sizeof(char *),
		sizeof(uint8_t),
		sizeof(uint16_t),
		sizeof(double),
		sizeof(uint32_t),
		sizeof(uint64_t),
		sizeof(float) * 2,
		sizeof(double) * 2,
		sizeof(uint16_t),
	};
	if((type > 0) && (type < (sizeof(typesz) / sizeof((typesz)[0]))))
		return typesz[type];
	return typesz[0];
}

struct onnx_tensor_t * onnx_tensor_search(struct onnx_context_t * ctx, const char * name)
{
	if(ctx)
		return hmap_search(ctx->map, name);
	return NULL;
}

int onnx_tensor_equal(struct onnx_tensor_t * a, struct onnx_tensor_t * b)
{
	size_t i;

	if(!a || !b)
		return 0;
	if(a->type != b->type)
		return 0;
	if(a->ndim != b->ndim)
		return 0;
	if(a->ndata != b->ndata)
		return 0;
	if(a->ndim > 0)
	{
		if(memcmp(a->dims, b->dims, sizeof(int) * a->ndim) != 0)
			return 0;
	}
	switch(a->type)
	{
	case ONNX_TENSOR_TYPE_BOOL:
	case ONNX_TENSOR_TYPE_INT8:
	case ONNX_TENSOR_TYPE_INT16:
	case ONNX_TENSOR_TYPE_INT32:
	case ONNX_TENSOR_TYPE_INT64:
	case ONNX_TENSOR_TYPE_UINT8:
	case ONNX_TENSOR_TYPE_UINT16:
	case ONNX_TENSOR_TYPE_UINT32:
	case ONNX_TENSOR_TYPE_UINT64:
		if(memcmp(a->datas, b->datas, a->ndata * onnx_tensor_type_sizeof(a->type)) != 0)
			return 0;
		break;
	case ONNX_TENSOR_TYPE_BFLOAT16:
		{
			uint16_t * p = (uint16_t *)a->datas;
			uint16_t * q = (uint16_t *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabsf(bfloat16_to_float32(p[i]) - bfloat16_to_float32(q[i])) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT16:
		{
			uint16_t * p = (uint16_t *)a->datas;
			uint16_t * q = (uint16_t *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabsf(float16_to_float32(p[i]) - float16_to_float32(q[i])) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT32:
		{
			float * p = (float *)a->datas;
			float * q = (float *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabsf(p[i] - q[i]) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_FLOAT64:
		{
			double * p = (double *)a->datas;
			double * q = (double *)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(fabs(p[i] - q[i]) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_COMPLEX64:
		{
			float * p = (float *)a->datas;
			float * q = (float *)b->datas;
			for(i = 0; i < a->ndata * 2; i++)
			{
				if(fabsf(p[i] - q[i]) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_COMPLEX128:
		{
			double * p = (double *)a->datas;
			double * q = (double *)b->datas;
			for(i = 0; i < a->ndata * 2; i++)
			{
				if(fabs(p[i] - q[i]) > 1e-3)
					return 0;
			}
		}
		break;
	case ONNX_TENSOR_TYPE_STRING:
		{
			char ** p = (char **)a->datas;
			char ** q = (char **)b->datas;
			for(i = 0; i < a->ndata; i++)
			{
				if(p[i] && q[i] && (strcmp(p[i], q[i]) != 0))
					return 0;
			}
		}
		break;
	default:
		break;
	}
	return 1;
}

void onnx_tensor_reinit(struct onnx_tensor_t * t, enum onnx_tensor_type_t type, int * dims, int ndim)
{
	char ** str;
	size_t n;
	int sz, i;

	if(t)
	{
		if(t->ndim > 0)
		{
			if(t->strides)
			{
				free(t->strides);
				t->strides = NULL;
			}
			if(t->dims)
			{
				free(t->dims);
				t->dims = NULL;
			}
			t->ndim = 0;
		}
		if((t->ndata > 0) && t->datas)
		{
			if(t->type == ONNX_TENSOR_TYPE_STRING)
			{
				str = (char **)t->datas;
				for(size_t idx = 0; idx < t->ndata; idx++)
				{
					if(str[idx])
					{
						free(str[idx]);
						str[idx] = NULL;
					}
				}
			}
			free(t->datas);
			t->datas = NULL;
			t->ndata = 0;
		}
		t->type = type;
		if(t->type != ONNX_TENSOR_TYPE_UNDEFINED)
		{
			if((ndim > 0) && dims)
			{
				for(i = 0; i < ndim; i++)
				{
					if(dims[i] <= 0)
						return;
				}
				t->strides = malloc(sizeof(int) * ndim);
				t->dims = malloc(sizeof(int) * ndim);
				if(t->strides && t->dims)
				{
					t->strides[ndim - 1] = 1;
					for(i = ndim - 2; i >= 0; i--)
						t->strides[i] = dims[i + 1] * t->strides[i + 1];
					memcpy(t->dims, dims, sizeof(int) * ndim);
					t->ndim = ndim;
					for(i = 0, n = 1; i < t->ndim; i++)
						n *= t->dims[i];
					sz = onnx_tensor_type_sizeof(t->type);
					if(sz > 0)
					{
						t->datas = malloc(n * sz);
						if(t->datas)
						{
							memset(t->datas, 0, n * sz);
							t->ndata = n;
						}
					}
				}
				else
				{
					if(t->strides)
					{
						free(t->strides);
						t->strides = NULL;
					}
					if(t->dims)
					{
						free(t->dims);
						t->dims = NULL;
					}
				}
			}
			else
			{
				sz = onnx_tensor_type_sizeof(t->type);
				if(sz > 0)
				{
					t->datas = malloc(sz);
					if(t->datas)
					{
						memset(t->datas, 0, sz);
						t->ndata = 1;
					}
				}
			}
		}
	}
}

void onnx_tensor_apply(struct onnx_tensor_t * t, void * buf, size_t len)
{
	size_t l;
	int sz;

	if(t)
	{
		if(t->datas && buf && (len > 0))
		{
			sz = onnx_tensor_type_sizeof(t->type);
			if(sz > 0)
			{
				if(t->type == ONNX_TENSOR_TYPE_STRING)
				{
					char ** p = (char **)t->datas;
					char ** q = (char **)buf;
					for(size_t idx = 0; idx < t->ndata; idx++)
					{
						if(p[idx])
						{
							free(p[idx]);
							p[idx] = NULL;
						}
					}
					l = min(t->ndata, (size_t)len);
					for(size_t idx = 0; idx < l; idx++)
						p[idx] = strdup(q[idx]);
				}
				else
				{
					l = t->ndata * sz;
					if(l > 0)
						memcpy(t->datas, buf, min(l, len));
				}
			}
		}
	}
}

static Onnx__AttributeProto * onnx_search_attribute(struct onnx_node_t * n, const char * name)
{
	Onnx__AttributeProto * attr;
	int i;

	if(n && name)
	{
		for(i = 0; i < n->proto->n_attribute; i++)
		{
			attr = n->proto->attribute[i];
			if(strcmp(attr->name, name) == 0)
				return attr;
		}
	}
	return NULL;
}

float onnx_attribute_read_float(struct onnx_node_t * n, const char * name, float def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT))
		return attr->f;
	return def;
}

int64_t onnx_attribute_read_int(struct onnx_node_t * n, const char * name, int64_t def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT))
		return attr->i;
	return def;
}

char * onnx_attribute_read_string(struct onnx_node_t * n, const char * name, char * def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING))
	{
		if(attr->s.len > 0)
		{
			attr->s.data[attr->s.len] = 0;
			return (char *)attr->s.data;
		}
	}
	return def;
}

int onnx_attribute_read_floats(struct onnx_node_t * n, const char * name, float ** floats)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS))
	{
		*floats = attr->floats;
		return attr->n_floats;
	}
	return 0;
}

int onnx_attribute_read_ints(struct onnx_node_t * n, const char * name, int64_t ** ints)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS))
	{
		*ints = attr->ints;
		return attr->n_ints;
	}
	return 0;
}

int onnx_attribute_read_tensor(struct onnx_node_t * n, const char * name, struct onnx_tensor_t * t)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);
	int * dims = NULL;
	int ndim = 0;
	int i;

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR))
	{
		if(attr->t)
		{
			if(attr->t->n_dims > 0)
			{
				dims = malloc(sizeof(int) * attr->t->n_dims);
				if(dims)
				{
					for(i = 0; i < attr->t->n_dims; i++)
						dims[i] = attr->t->dims[i];
					ndim = attr->t->n_dims;
				}
			}
			if((t->ndim != ndim) || (memcmp(t->dims, dims, sizeof(int) * ndim) != 0) || (t->type != (enum onnx_tensor_type_t)attr->t->data_type))
				onnx_tensor_reinit(t, (enum onnx_tensor_type_t)attr->t->data_type, dims, ndim);
			if((ndim > 0) && dims)
				free(dims);
			onnx_tensor_copy_from_tensor_proto(t, attr->t);
			return 1;
		}
	}
	return 0;
}

Onnx__GraphProto * onnx_attribute_read_graph(struct onnx_node_t * n, const char * name, Onnx__GraphProto * def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH))
	{
		if(attr->g)
			return attr->g;
	}
	return def;
}

Onnx__SparseTensorProto * onnx_attribute_read_sparse_tensor(struct onnx_node_t * n, const char * name, Onnx__SparseTensorProto * def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR))
	{
		if(attr->sparse_tensor)
			return attr->sparse_tensor;
	}
	return def;
}