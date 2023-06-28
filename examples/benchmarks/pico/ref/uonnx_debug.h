#ifndef __UONNX_DEBUG_H__
#define __UONNX_DEBUG_H__

#ifdef __cplusplus
extern "C" {
#endif


#include "uonnx.h"

/**
 * @brief Check if ptr1 and ptr2 points to the same address.
 * 
 * @param ptr1 
 * @param ptr2 
 */
void addrcmp(void *ptr1, void *ptr2);

/**
 * @brief Get the cpu heap usage using mallinfo
 * 
 * @param TAG 
 */
void get_cpu_heap(const char * TAG);

void dump_plannerproto(PlannerProto * planner);

/**
 * @brief Dump node n
 * 
 * @param n 
 */
void dump_node(Node * n);

/**
 * @brief Dump tensor t
 * 
 * @param t 
 */
void dump_tensor(Tensor * t);

/**
 * @brief Dump TensorArena.
 * 
 * @param arena to dump
 * @param type TensorType of datas
 * @param n number of datas
 */
void dump_arena(TensorArena * arena, TensorType type, size_t n);

/**
 * @brief Dump Graph g
 * 
 * @param g 
 */
void dump_graph(Graph * g);

/**
 * @brief Dump planner
 * 
 * @param planner 
 */
void dump_planner(Planner * planner);

#ifdef __cplusplus
}
#endif

#endif

