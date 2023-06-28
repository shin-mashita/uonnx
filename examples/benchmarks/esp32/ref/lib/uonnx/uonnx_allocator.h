#ifndef __UONNX_ALLOCATOR_H__
#define __UONNX_ALLOCATOR_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "uonnx.h"

/**
 * @brief Alloc space for tensor. Data set to 0.
 * 
 * @param tensor_id 
 * @param type 
 * @param dims 
 * @param ndim 
 * @param isInitializer 
 * @return Tensor* 
 */
Tensor * tensor_alloc_nodatas(uint32_t tensor_id, TensorType type, int * dims, int ndim, uint8_t isInitializer);

/**
 * @brief Initialize uONNX graph from PlannerProto
 * 
 * @param model 
 * @param planner 
 * @param arena 
 * @return Graph* 
 */
Graph * graph_init_from_PlannerProto(ModelProto * model, PlannerProto * planner, TensorArena * arena);

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
 * @brief Free memory allocated from Graph
 * 
 * @param g Graph to free
 */
void free_graph(Graph * g); 

#ifdef __cplusplus
}
#endif

#endif