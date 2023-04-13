
#ifndef __ONNX_RESOLVER_H__
#define __ONNX_RESOLVER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "onnx_config.h"
#include "onnx_dtypes.h"

struct onnx_resolver_t {
	const char * name;

	void * (*create)(void);
	void (*destroy)(void * rctx);

    void (*op_Relu)(struct onnx_node_t * n);
	void (*op_Conv)(struct onnx_node_t * n);
	void (*op_Reshape)(struct onnx_node_t * n);
	void (*op_Add)(struct onnx_node_t * n);
	void (*op_MatMul)(struct onnx_node_t * n);
	void (*op_MaxPool)(struct onnx_node_t * n);
    };

void resolver_solve_operator(struct onnx_resolver_t * r, struct onnx_node_t * n);

void * resolver_default_create(void);
void resolver_default_destroy(void * rctx);

void resolver_default_op_Relu(struct onnx_node_t * n);
void resolver_default_op_Conv(struct onnx_node_t * n);
void resolver_default_op_Reshape(struct onnx_node_t * n);
void resolver_default_op_Add(struct onnx_node_t * n);
void resolver_default_op_MatMul(struct onnx_node_t * n);
void resolver_default_op_MaxPool(struct onnx_node_t * n);

extern struct onnx_resolver_t resolver_default;

#ifdef __cplusplus
}
#endif
#endif