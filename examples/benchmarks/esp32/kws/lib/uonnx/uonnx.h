#ifndef __UONNX_H__
#define __UONNX_H__

#ifdef __cplusplus
extern "C" {
#endif

/* UONNX DEFINES HERE */
#define UONNX_OPS_ABS
#define UONNX_OPS_ADD
#define UONNX_DTYPE_FP32
#define UONNX_DTYPE_FP16 // TODO: add more defines here for conditional compilation

/* CORES */
#include "uonnx_config.h"
#include "proto/onnx.proto3.pb-c.h"
#include "proto/planner.proto3.pb-c.h"
#include "uonnx_dtypes.h"

#include "uonnx_planner.h"
#include "uonnx_allocator.h"
#include "uonnx_arena.h"
#include "uonnx_debug.h"
#include "uonnx_loader.h"
#include "uonnx_resolver.h"
#include "uonnx_utils.h"

Context * uonnx_init(const void * model_buf, size_t model_len, const void * planner_buf, size_t planner_len);
void uonnx_run(Context * ctx);
void uonnx_free(Context * ctx);

#ifdef __cplusplus
}
#endif

#endif