#ifndef __UONNX_H__
#define __UONNX_H__

#ifdef __cplusplus
extern "C" {
#endif

/* UONNX DEFINES HERE */
#define UONNX_OPS_ABS
#define UONNX_OPS_ACOS
#define UONNX_OPS_ACOSH
#define UONNX_OPS_ADD
#define UONNX_OPS_AND
#define UONNX_OPS_ARGMAX
#define UONNX_OPS_ARGMIN
#define UONNX_OPS_ASIN
#define UONNX_OPS_ASINH
#define UONNX_OPS_ATAN
#define UONNX_OPS_ATANH
#define UONNX_OPS_AVERAGEPOOL
#define UONNX_OPS_CEIL
#define UONNX_OPS_CELU
#define UONNX_OPS_CONV
#define UONNX_OPS_COS
#define UONNX_OPS_COSH
#define UONNX_OPS_ELU
#define UONNX_OPS_EQUAL
#define UONNX_OPS_ERF
#define UONNX_OPS_EXP
#define UONNX_OPS_FLOOR
#define UONNX_OPS_GEMM
#define UONNX_OPS_GLOBALAVERAGEPOOL
#define UONNX_OPS_GLOBALLPPOOL
#define UONNX_OPS_GLOBALMAXPOOL
#define UONNX_OPS_GREATER
#define UONNX_OPS_GREATEROREQUAL
#define UONNX_OPS_HARDSIGMOID
#define UONNX_OPS_LEAKYRELU
#define UONNX_OPS_LESS
#define UONNX_OPS_LESSOREQUAL
#define UONNX_OPS_LOG
#define UONNX_OPS_LOGSOFTMAX
#define UONNX_OPS_MATMUL
#define UONNX_OPS_MAX
#define UONNX_OPS_MAXPOOL
#define UONNX_OPS_MEAN
#define UONNX_OPS_MIN
#define UONNX_OPS_MOD
#define UONNX_OPS_MUL
#define UONNX_OPS_NEG
#define UONNX_OPS_POW
#define UONNX_OPS_PRELU
#define UONNX_OPS_RELU
#define UONNX_OPS_RESHAPE
#define UONNX_OPS_SELU
#define UONNX_OPS_SIGMOID
#define UONNX_OPS_SIGN
#define UONNX_OPS_SIN
#define UONNX_OPS_SINH
#define UONNX_OPS_SOFTMAX
#define UONNX_OPS_SUB
#define UONNX_OPS_SUM
#define UONNX_OPS_TAN
#define UONNX_OPS_TANH
#define UONNX_OPS_TRANSPOSE

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