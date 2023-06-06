#ifndef __UONNX_UTILS_H__
#define __UONNX_UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <uonnx.h>

char * TensorType2String(TensorType t);
int onnx_tensor_type_sizeof(TensorType type);

static inline int onnx_tensor_is_scalar(Tensor * t)
{
    return ((t->ndim == 0) && (t->ndata == 1)) ? 1 : 0;
}

static inline int onnx_tensor_broadcast_is_valid(Tensor * x, int * dims, int ndim)
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

static inline int onnx_tensor_indices_to_offset(Tensor * t, int * indices)
{
    int offset, i;

    for(i = 0, offset = 0; i < t->ndim; i++)
        offset += indices[i] * t->strides[i];
    return offset;
}

static inline void onnx_tensor_offset_to_indices(Tensor * t, int offset, int * indices)
{
    int i;

    for(i = t->ndim - 1; i >= 0; i--)
    {
        indices[i] = offset % t->dims[i];
        offset /= t->dims[i];
    }
}

static inline int onnx_tensor_reshape(Tensor * y, int * dims, int ndim, TensorType type)
{
    if((y->ndim != ndim) || (dims && (memcmp(y->dims, dims, sizeof(int) * y->ndim) != 0)) || (y->type != type))
    {
        // Conditional compile
        // printf("Tensor %s invalid and need to be reinitialized.\n", y->name);
        return 0;
    }
        
    return 1;
}

static inline int onnx_tensor_reshape_identity(Tensor * y, Tensor * x, TensorType type)
{
    if((y->ndim != x->ndim) || (memcmp(y->dims, x->dims, sizeof(int) * y->ndim) != 0) || (y->type != type))
    {
        // Conditional compile
        // printf("Tensor %s invalid and need to be reinitialized.\n", y->name);
        return 0;
    }
    return 1;
}

static inline int onnx_tensor_reshape_multi_broadcast(Tensor * y, Tensor * a, Tensor * b, TensorType type)
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
    {
        // Conditional compile
        // printf("Tensor %s invalid and need to be reinitialized.\n", y->name);
        return 0;
    }
    return 1;
}

static inline void * onnx_tensor_broadcast_map_address(Tensor * x, Tensor * y, int offset)
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

float onnx_attribute_read_float(Node * n, const char * name, float def);
int64_t onnx_attribute_read_int(Node * n, const char * name, int64_t def);
char * onnx_attribute_read_string(Node * n, const char * name, char * def);
int onnx_attribute_read_ints(Node * n, const char * name, int64_t ** ints);
int onnx_attribute_read_floats(Node * n, const char * name, float ** floats);
// int onnx_attribute_read_tensor(Node * n, const char * name, Tensor * t);
Onnx__GraphProto * onnx_attribute_read_graph(Node * n, const char * name, Onnx__GraphProto * def);
Onnx__SparseTensorProto * onnx_attribute_read_sparse_tensor(Node * n, const char * name, Onnx__SparseTensorProto * def);

#ifdef __cplusplus
}
#endif

#endif