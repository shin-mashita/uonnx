#ifndef __ONNX_ALLOCATOR_H__
#define __ONNX_ALLOCATOR_H__

#include "onnx_config.h"
#include "onnx_resolver.h"
#include "onnx_logger.h"
#include "proto/onnx.proto3.pb-c.h"
#include "onnx_dtypes.h"

struct onnx_context_t * onnx_context_alloc(const void * buf, size_t len, struct onnx_resolver_t ** r, int rlen);
struct onnx_context_t * onnx_context_alloc_from_file(const char * filename, struct onnx_resolver_t ** r, int rlen);
void onnx_context_free(struct onnx_context_t * ctx);

struct onnx_graph_t * onnx_graph_alloc(struct onnx_context_t * ctx, Onnx__GraphProto * graph);
void onnx_graph_free(struct onnx_graph_t * g);

struct onnx_tensor_t * onnx_tensor_alloc(const char * name, enum onnx_tensor_type_t type, int * dims, int ndim);
struct onnx_tensor_t * onnx_tensor_alloc_from_file(const char * filename);
void onnx_tensor_free(struct onnx_tensor_t * t);


#endif