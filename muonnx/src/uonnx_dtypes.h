#ifndef __UONNX_DTYPES_H__
#define __UONNX_DTYPES_H__

#include <uonnx.h>

typedef Onnx__ModelProto ModelProto;
typedef Onnx__GraphProto GraphProto; 
typedef Onnx__TensorProto TensorProto;

typedef enum TensorType
{
	TENSOR_TYPE_UNDEFINED	= 0,
	TENSOR_TYPE_BOOL		= 9,
	TENSOR_TYPE_INT8		= 3,
	TENSOR_TYPE_INT16		= 5,
	TENSOR_TYPE_INT32		= 6,
	TENSOR_TYPE_INT64		= 7,
	TENSOR_TYPE_UINT8		= 2,
	TENSOR_TYPE_UINT16		= 4,
	TENSOR_TYPE_UINT32		= 12,
	TENSOR_TYPE_UINT64		= 13,
	TENSOR_TYPE_BFLOAT16	= 16,
	TENSOR_TYPE_FLOAT16		= 10,
	TENSOR_TYPE_FLOAT32		= 1,
	TENSOR_TYPE_FLOAT64		= 11,
	TENSOR_TYPE_COMPLEX64	= 14,
	TENSOR_TYPE_COMPLEX128	= 15,
	TENSOR_TYPE_STRING		= 8,
} TensorType;

typedef struct Tensor
{
    char * name;
    TensorType type;
    void * datas;
    size_t ndata;
    int64_t * dims;
    size_t ndim;
	uint8_t isProto; // May remove later
} Tensor;

typedef struct Node
{
    int opset;
	Tensor ** inputs;
	int ninputs;
	Tensor ** outputs;
	int noutputs;

} Node;

typedef struct Graph
{
    Node * nodes;
    int nlen;
} Graph;

typedef struct TensorArena //TODO: Define in separate C module
{
	Tensor ** tensors;
	int n_tensors;
	void * datas;
} TensorArena;

typedef struct Resolver //TODO: Define in another C module
{
	size_t rlen;
} Resolver;

typedef struct Context 
{
    ModelProto * model; // to free after context initialization
    // Resolver * resolver;
    Graph * graph;
} Context;




#endif