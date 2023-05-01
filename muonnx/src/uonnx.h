#ifndef __UONNX_H__
#define __UONNX_H__

#define min(a, b)           ({typeof(a) _amin = (a); typeof(b) _bmin = (b); (void)(&_amin == &_bmin); _amin < _bmin ? _amin : _bmin;})
#define max(a, b)           ({typeof(a) _amax = (a); typeof(b) _bmax = (b); (void)(&_amax == &_bmax); _amax > _bmax ? _amax : _bmax;})
#define clamp(v, a, b)      min(max(a, v), b)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <malloc.h>
#include <float.h>
#include <math.h>

#include "proto/onnx.proto3.pb-c.h"
#include "uonnx_dtypes.h"
#include "uonnx_utils.h"
#include "uonnx_debug.h"
// #include "onnx_allocator.h"
// #include "onnx_dtypes.h"
// #include "onnx_loader.h"
// #include "onnx_logger.h"
// #include "onnx_resolver.h"
// #include "onnx_utils.h"

#endif