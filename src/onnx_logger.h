#ifndef __ONNX_LOGGER_H__
#define __ONNX_LOGGER_H__

#define ONNX_LOG(...) (printf("[uONNX] "), printf(__VA_ARGS__))

#include "onnx_config.h"
#include "proto/onnx.proto3.pb-c.h"
#include "onnx_dtypes.h"

void onnx_model_dump(Onnx__ModelProto * model);

void onnx_tensor_dump(struct onnx_tensor_t * t, int detail);
void onnx_node_dump(struct onnx_node_t * n, int detail);
void onnx_graph_dump(struct onnx_graph_t * g, int detail);
void onnx_context_dump(struct onnx_context_t * ctx, int detail);

#endif