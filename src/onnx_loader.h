#ifndef __ONNX_LOADER_H__
#define __ONNX_LOADER_H__

#include "onnx_config.h"
#include "proto/onnx.proto3.pb-c.h"

Onnx__ModelProto * onnx_load_model(const void * buf, size_t len);
Onnx__ModelProto * onnx_load_model_from_file(const char * filename);




#endif