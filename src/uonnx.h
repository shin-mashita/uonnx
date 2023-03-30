#ifndef __UONNX_H__
#define __UONNX_H__

#include "onnx_allocator.h"
#include "onnx_dtypes.h"
#include "onnx_loader.h"
#include "onnx_logger.h"
#include "onnx_resolver.h"
#include "onnx_utils.h"

void onnx_run(struct onnx_context_t * ctx);

#endif