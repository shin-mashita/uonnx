#ifndef __UONNX_H__
#define __UONNX_H__

#include "onnx_allocator.h"
#include "onnx_dtypes.h"
#include "onnx_loader.h"
#include "onnx_logger.h"
#include "onnx_resolver.h"
#include "onnx_utils.h"

// Default resolvers here. Comment out to use custom resolvers.
#include "onnx_default_resolver.h"

// Custom resolver here. Uncomment to use. 
// #include "onnx_custom_resolver.h"

void onnx_run(struct onnx_context_t * ctx);

#endif