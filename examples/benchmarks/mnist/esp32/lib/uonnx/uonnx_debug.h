#ifndef __UONNX_DEBUG_H__
#define __UONNX_DEBUG_H__

#include <uonnx.h>

void dump_tensor(Tensor * t);
void dump_arena(TensorArena * arena, TensorType type, size_t n);
void dump_graph(Graph * g);

#endif