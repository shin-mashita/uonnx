#ifndef __UONNX_ARENA_H__
#define __UONNX_ARENA_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "uonnx.h"


/**
 * @brief Initialize buffer for tensors.
 * 
 * @param MAX_TENSORS indicates maximum tensor count the arena can carry
 * @param MAX_BYTES indicates maximum bytes allocated for tensor data
 * @return TensorArena* 
 */
TensorArena * arena_init(const int MAX_TENSORS, const size_t MAX_BYTES);

/**
 * @brief Free TensorArena
 * 
 * @param arena to free
 */
void free_arena(TensorArena * arena);

/**
 * @brief Add tensor to arena at index arena_pos
 * 
 * @param tensor 
 * @param arena 
 * @param arena_pos 
 */
void arena_add_tensor(Tensor * tensor, TensorArena * arena, int arena_pos);

/**
 * @brief Add initializer to arena.
 * 
 * @param initializer 
 * @param arena 
 */
void arena_add_initializer(TensorProto * initializer, TensorArena * arena);

/**
 * @brief Add intermediate tensor(from plan) to arena.
 * 
 * @param plan 
 * @param arena 
 */
void arena_add_intermediate(PlanProto * plan, TensorArena * arena);

/**
 * @brief Search tensors in arena by name
 * 
 * @param arena Arena
 * @param name Name of tensor
 * @return Tensor* 
 */
Tensor * tensor_search(TensorArena * arena, const char * name);

#ifdef __cplusplus
}
#endif

#endif