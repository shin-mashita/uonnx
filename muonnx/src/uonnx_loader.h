#ifndef __UONNX_LOADER_H__
#define __UONNX_LOADER_H__

#include <uonnx.h>

Onnx__ModelProto * load_model_buf(const void * buf, size_t len);
Onnx__ModelProto * load_model(const char * filename);
void free_model(ModelProto * model);

#endif