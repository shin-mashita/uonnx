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

Context * uonnx_init(   const char * filename, 
                        const void * model_buf,
                        size_t model_len,
                        void * input_buf, 
                        size_t input_len,
                        char * input_name, 
                        void * output_buf, 
                        size_t output_len,
                        char * output_name, 
                        Planner * planner);

void uonnx_run(Context * context);
void uonnx_free(Context * context);

#ifdef __cplusplus
}
#endif

#endif