#ifndef __UONNX_ALLOCATOR_H__
#define __UONNX_ALLOCATOR_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <uonnx.h>

Tensor * tensor_alloc_nodatas(const char * name, TensorType type, int * dims, int ndim, uint8_t isInitializer);
void arena_add_tensor(Tensor * tensor, TensorArena * arena, int arena_pos);
void arena_add_initializer(TensorProto * initializer, TensorArena * arena);
void arena_add_intermediate(PlanProto * plan, TensorArena * arena);
Graph * graph_init_from_PlannerProto(GraphProto * gproto, ModelProto * model, PlannerProto * planner, TensorArena * arena);

/**
 * @brief Add/initialize tensor in arena
 * 
 * @param name Tensor name
 * @param type Tensor type 
 * @param dims Tensor dimensions
 * @param ndim Count of dimensions
 * @param arena Address of TensorArena.
 * @param tensor Tensor object to add. If NULL, malloc instead.
 * @param data_idx Start index of tensor datas in arena. Required for arenas and must be multiplied by sizeof(type).
 * @return Tensor*
 */
Tensor * tensor_init(const char *name, TensorType type, int *dims, int ndim, TensorArena *arena, Tensor * tensor, int data_idx);

/**
 * @brief Add/Initialize tensor to arena from model ValueInfoProto
 * 
 * @param v Source ValueInfoProto.
 * @param arena Destination arena.
 * @param tensor Tensor object to situate infos from ValueInfoProto
 * @param data_idx Start index of tensor datas in arena
 * @return Tensor* 
 */
Tensor * tensor_init_from_value_info(ValueInfoProto * v, TensorArena * arena, Tensor * tensor, int data_idx);

/**
 * @brief Add/Initialize tensor from proto. Does not memcpy. tensor->datas directly points to ModelProto. Used for initializers.
 * 
 * @param tp Source TensorProto from model
 * @param tensor  Tensor object
 * @return Tensor* 
 */
Tensor * tensor_init_from_proto(TensorProto * tp, Tensor * tensor);

/**
 * @brief Free malloc'ed Tensor object
 * 
 * @param t Tensor to free
 */
void free_tensor(Tensor * t);

/**
 * @brief Remove Tensor object from arena
 * 
 * @param t Tensor to free
 */
void free_tensor_from_arena(Tensor * t);

/**
 * @brief Set values in Tensor
 * 
 * @param datas source datas to assign
 * @param size size of datas
 * @param t dest tensor
 */
void tensor_apply(void * datas, size_t size, Tensor * t);

/**
 * @brief Initialize Graph object from GraphProto
 * 
 * @param gproto GraphProto from model
 * @param model Loaded ModelProto
 * @param arena TensorArena
 * @param planner Memory planner
 * @return Graph* 
 */
Graph * graph_init(GraphProto * gproto, ModelProto * model, TensorArena * arena, Planner * planner);

/**
 * @brief Free memory allocated from Graph
 * 
 * @param g Graph to free
 */
void free_graph(Graph * g); 

#ifdef __cplusplus
}
#endif

#endif