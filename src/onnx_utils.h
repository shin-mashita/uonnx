#ifndef __ONNX_UTILS_H__
#define __ONNX_UTILS_H__

#include "onnx_config.h"
#include "proto/onnx.proto3.pb-c.h"
#include "onnx_dtypes.h"

/* Mapping of tensor type */
const char * onnx_tensor_type_tostring(enum onnx_tensor_type_t type);

/* Get tensor size */
int onnx_tensor_type_sizeof(enum onnx_tensor_type_t type);

/* Search tensor using name within context */
struct onnx_tensor_t * onnx_tensor_search(struct onnx_context_t * ctx, const char * name);

/* Check if tensors are equal */
int onnx_tensor_equal(struct onnx_tensor_t * a, struct onnx_tensor_t * b);

/* Reinitialize a tensor */
void onnx_tensor_reinit(struct onnx_tensor_t * t, enum onnx_tensor_type_t type, int * dims, int ndim);

/* Apply values to a tensor */
void onnx_tensor_apply(struct onnx_tensor_t * t, void * buf, size_t len);


static inline int onnx_tensor_is_scalar(struct onnx_tensor_t * t)
{
	return ((t->ndim == 0) && (t->ndata == 1)) ? 1 : 0;
}

static inline int onnx_tensor_broadcast_is_valid(struct onnx_tensor_t * x, int * dims, int ndim)
{
	int i;

	if(x->ndim > ndim)
		return 0;
	for(i = 1; i <= x->ndim; i++)
	{
		if((x->dims[x->ndim - i] != 1) && (x->dims[x->ndim - i] != dims[ndim - i]))
			return 0;
	}
	return 1;
}

static inline int onnx_tensor_indices_to_offset(struct onnx_tensor_t * t, int * indices)
{
	int offset, i;

	for(i = 0, offset = 0; i < t->ndim; i++)
		offset += indices[i] * t->strides[i];
	return offset;
}

static inline void onnx_tensor_offset_to_indices(struct onnx_tensor_t * t, int offset, int * indices)
{
	int i;

	for(i = t->ndim - 1; i >= 0; i--)
	{
		indices[i] = offset % t->dims[i];
		offset /= t->dims[i];
	}
}

static inline int onnx_tensor_reshape(struct onnx_tensor_t * y, int * dims, int ndim, enum onnx_tensor_type_t type)
{
	if((y->ndim != ndim) || (dims && (memcmp(y->dims, dims, sizeof(int) * y->ndim) != 0)) || (y->type != type))
	{
		printf("Tensor reinitializing.");
		onnx_tensor_reinit(y, type, dims, ndim);
	}
		
	return 1;
}

static inline int onnx_tensor_reshape_identity(struct onnx_tensor_t * y, struct onnx_tensor_t * x, enum onnx_tensor_type_t type)
{
	if((y->ndim != x->ndim) || (memcmp(y->dims, x->dims, sizeof(int) * y->ndim) != 0) || (y->type != type))
		printf("Tensor reinitializing."); onnx_tensor_reinit(y, type, x->dims, x->ndim);
	return 1;
}

static inline int onnx_tensor_reshape_multi_broadcast(struct onnx_tensor_t * y, struct onnx_tensor_t * a, struct onnx_tensor_t * b, enum onnx_tensor_type_t type)
{
	int ndim = max(a->ndim, b->ndim);
	int dims[ndim];
	int i, j, k;

	if(ndim > 0)
	{
		for(i = a->ndim - 1, j = b->ndim - 1, k = ndim - 1; k >= 0; k--)
		{
			if(i < 0)
				dims[k] = b->dims[j--];
			else if(j < 0)
				dims[k] = a->dims[i--];
			else
			{
				if(a->dims[i] == b->dims[j])
					dims[k] = a->dims[i];
				else if((a->dims[i] == 1) || (b->dims[j] == 1))
					dims[k] = (a->dims[i] > b->dims[j]) ? a->dims[i] : b->dims[j];
				else
					return 0;
				i--;
				j--;
			}
		}
	}
	if((y->type != type) || (y->ndim != ndim) || (memcmp(y->dims, dims, sizeof(int) * ndim) != 0))
		printf("Tensor reinitializing."); onnx_tensor_reinit(y, type, dims, ndim);
	return 1;
}

static inline void * onnx_tensor_broadcast_map_address(struct onnx_tensor_t * x, struct onnx_tensor_t * y, int offset)
{
	int xndim = x->ndim;
	int yndim = y->ndim;

	if((xndim > 0) && (yndim > 0))
	{
		int dndim = yndim - xndim;
		int ix[xndim];
		int iy[yndim];
		int i;

		onnx_tensor_offset_to_indices(y, offset, iy);
		for(i = 0; i < xndim; i++)
			ix[i] = iy[dndim + i] % x->dims[i];
		return x->datas + onnx_tensor_indices_to_offset(x, ix) * onnx_tensor_type_sizeof(x->type);
	}
	return x->datas;
}

float onnx_attribute_read_float(struct onnx_node_t * n, const char * name, float def);
int64_t onnx_attribute_read_int(struct onnx_node_t * n, const char * name, int64_t def);
char * onnx_attribute_read_string(struct onnx_node_t * n, const char * name, char * def);
int onnx_attribute_read_ints(struct onnx_node_t * n, const char * name, int64_t ** ints);
int onnx_attribute_read_floats(struct onnx_node_t * n, const char * name, float ** floats);
int onnx_attribute_read_tensor(struct onnx_node_t * n, const char * name, struct onnx_tensor_t * t);
Onnx__GraphProto * onnx_attribute_read_graph(struct onnx_node_t * n, const char * name, Onnx__GraphProto * def);
Onnx__SparseTensorProto * onnx_attribute_read_sparse_tensor(struct onnx_node_t * n, const char * name, Onnx__SparseTensorProto * def);

#endif