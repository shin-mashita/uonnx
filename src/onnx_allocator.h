#ifndef __ONNX_ALLOCATOR_H__
#define __ONNX_ALLOCATOR_H__

#include "onnxconf.h"
#include "proto/onnx.proto3.pb-c.h"

enum onnx_tensor_type_t {
	ONNX_TENSOR_TYPE_UNDEFINED	= 0,
	ONNX_TENSOR_TYPE_BOOL		= 9,
	ONNX_TENSOR_TYPE_INT8		= 3,
	ONNX_TENSOR_TYPE_INT16		= 5,
	ONNX_TENSOR_TYPE_INT32		= 6,
	ONNX_TENSOR_TYPE_INT64		= 7,
	ONNX_TENSOR_TYPE_UINT8		= 2,
	ONNX_TENSOR_TYPE_UINT16		= 4,
	ONNX_TENSOR_TYPE_UINT32		= 12,
	ONNX_TENSOR_TYPE_UINT64		= 13,
	ONNX_TENSOR_TYPE_BFLOAT16	= 16,
	ONNX_TENSOR_TYPE_FLOAT16	= 10,
	ONNX_TENSOR_TYPE_FLOAT32	= 1,
	ONNX_TENSOR_TYPE_FLOAT64	= 11,
	ONNX_TENSOR_TYPE_COMPLEX64	= 14,
	ONNX_TENSOR_TYPE_COMPLEX128	= 15,
	ONNX_TENSOR_TYPE_STRING		= 8,
};

struct onnx_tensor_t {
	char * name;
	enum onnx_tensor_type_t type;
	int * strides;
	int * dims;
	int ndim;
	void * datas;
	size_t ndata;
};

struct onnx_node_t {
	struct onnx_context_t * ctx;
	struct onnx_resolver_t * r;
	void * rctx;
	int opset;
	struct onnx_tensor_t ** inputs;
	int ninput;
	struct onnx_tensor_t ** outputs;
	int noutput;
	Onnx__NodeProto * proto;

	int (*init)(struct onnx_node_t * n);
	int (*exit)(struct onnx_node_t * n);
	int (*reshape)(struct onnx_node_t * n);
	void (*operator)(struct onnx_node_t * n);
	void * priv;
};

struct onnx_graph_t {
	struct onnx_node_t * nodes;
	int nlen;
};

struct onnx_context_t {
	Onnx__ModelProto * model;
	struct hmap_t * map;
	struct onnx_resolver_t ** r;
	void ** rctx;
	int rlen;
	struct onnx_graph_t * g;
};

struct onnx_context_t * onnx_context_alloc(const void * buf, size_t len, struct onnx_resolver_t ** r, int rlen);
struct onnx_context_t * onnx_context_alloc_from_file(const char * filename, struct onnx_resolver_t ** r, int rlen);
void onnx_context_free(struct onnx_context_t * ctx);

struct onnx_graph_t * onnx_graph_alloc(struct onnx_context_t * ctx, Onnx__GraphProto * graph);
void onnx_graph_free(struct onnx_graph_t * g);

struct onnx_tensor_t * onnx_tensor_alloc(const char * name, enum onnx_tensor_type_t type, int * dims, int ndim);
struct onnx_tensor_t * onnx_tensor_alloc_from_file(const char * filename);
void onnx_tensor_free(struct onnx_tensor_t * t);


#endif