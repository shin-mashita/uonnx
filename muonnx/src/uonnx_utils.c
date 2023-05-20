#include "uonnx_utils.h"

// Extras
char * TensorType2String(TensorType t) 
{
    switch (t)
    {
        case TENSOR_TYPE_UNDEFINED	:
            return "UNDEFINED";
            break;
        case TENSOR_TYPE_BOOL		:
            return "BOOL";
            break;
        case TENSOR_TYPE_INT8		:
            return "INT8";
            break;
        case TENSOR_TYPE_INT16		:
            return "INT16";
            break;
        case TENSOR_TYPE_INT32		:
            return "INT32";
            break;
        case TENSOR_TYPE_INT64		:
            return "INT64";
            break;
        case TENSOR_TYPE_UINT8		:
            return "UINT8";
            break;
        case TENSOR_TYPE_UINT16		:
            return "UINT16";
            break;
        case TENSOR_TYPE_UINT32		:
            return "UINT32";
            break;
        case TENSOR_TYPE_UINT64		:
            return "UINT64";
            break;
        case TENSOR_TYPE_BFLOAT16	:
            return "BFLOAT16";
            break;
        case TENSOR_TYPE_FLOAT16	:
            return "FLOAT16";
            break;	
        case TENSOR_TYPE_FLOAT32	:
            return "FLOAT32";
            break;	
        case TENSOR_TYPE_FLOAT64	:
            return "FLOAT64";
            break;
        case TENSOR_TYPE_COMPLEX64	:
            return "COMPLEX64";
            break;
        case TENSOR_TYPE_COMPLEX128	:
            return "COMPLEX128";
            break;
        case TENSOR_TYPE_STRING		:
            return "STRING";
            break;
        default:
            return NULL;
            break;
    }
}


int onnx_tensor_type_sizeof(TensorType type)
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

static Onnx__AttributeProto * onnx_search_attribute(Node * n, const char * name)
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

float onnx_attribute_read_float(Node * n, const char * name, float def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT))
		return attr->f;
	return def;
}

int64_t onnx_attribute_read_int(Node * n, const char * name, int64_t def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT))
		return attr->i;
	return def;
}

char * onnx_attribute_read_string(Node * n, const char * name, char * def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING))
	{
		if(attr->s.len > 0)
		{
			attr->s.data[attr->s.len] = 0; // ERROR: Causing invalid write of size 1. Writing at end of index.
			return (char *)attr->s.data;
		}
	}
	return def;
}

int onnx_attribute_read_floats(Node * n, const char * name, float ** floats)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS))
	{
		*floats = attr->floats;
		return attr->n_floats;
	}
	return 0;
}

int onnx_attribute_read_ints(Node * n, const char * name, int64_t ** ints)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS))
	{
		*ints = attr->ints;
		return attr->n_ints;
	}
	return 0;
}

// USED IN Constant.c

// int onnx_attribute_read_tensor(Node * n, const char * name, Tensor * t)
// {
// 	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);
// 	int * dims = NULL;
// 	int ndim = 0;
// 	int i;

// 	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR))
// 	{
// 		if(attr->t)
// 		{
// 			if(attr->t->n_dims > 0)
// 			{
// 				dims = malloc(sizeof(int) * attr->t->n_dims);
// 				if(dims)
// 				{
// 					for(i = 0; i < attr->t->n_dims; i++)
// 						dims[i] = attr->t->dims[i];
// 					ndim = attr->t->n_dims;
// 				}
// 			}
// 			if((t->ndim != ndim) || (memcmp(t->dims, dims, sizeof(int) * ndim) != 0) || (t->type != (TensorType)attr->t->data_type))
// 				onnx_tensor_reinit(t, (TensorType)attr->t->data_type, dims, ndim);
// 			if((ndim > 0) && dims)
// 				free(dims);
// 			onnx_tensor_copy_from_tensor_proto(t, attr->t);
// 			return 1;
// 		}
// 	}
// 	return 0;
// }

Onnx__GraphProto * onnx_attribute_read_graph(Node * n, const char * name, Onnx__GraphProto * def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH))
	{
		if(attr->g)
			return attr->g;
	}
	return def;
}

Onnx__SparseTensorProto * onnx_attribute_read_sparse_tensor(Node * n, const char * name, Onnx__SparseTensorProto * def)
{
	Onnx__AttributeProto * attr = onnx_search_attribute(n, name);

	if(attr && (attr->type == ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR))
	{
		if(attr->sparse_tensor)
			return attr->sparse_tensor;
	}
	return def;
}

