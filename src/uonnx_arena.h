#ifndef __UONNX_ARENA_H__
#define __UONNX_ARENA_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <uonnx.h>

/**
 * @brief Initialize buffer for tensors.
 * 
 * @param MAX_TENSORS indicates maximum tensor count the arena can carry
 * @param MAX_BYTES indicates maximum bytes allocated for tensor data
 * @return TensorArena* 
 */
TensorArena * arena_init(const int MAX_TENSORS, const size_t MAX_BYTES);

/**
 * @brief Initialize TensorArena from Planner
 * 
 * @param planner Memory planner
 * @return TensorArena* 
 */
TensorArena * arena_init_from_planner(Planner * planner);

/**
 * @brief Free TensorArena
 * 
 * @param arena to free
 */
void free_arena(TensorArena * arena);

/**
 * @brief Search tensors in arena by name
 * 
 * @param arena Arena
 * @param name Name of tensor
 * @return Tensor* 
 */
Tensor * tensor_search(TensorArena * arena, char * name);

#ifdef __cplusplus
}
#endif

#endif