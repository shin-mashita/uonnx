#ifndef __UONNX_PLANNER_H__
#define __UONNX_PLANNER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <uonnx.h>

PlannerProto * load_planner_buf(const void * buf, size_t len);
PlannerProto * load_planner(const char * filename);
void free_plannerproto(PlannerProto * planner);

/**
 * @brief Initialize a memory planner
 * 
 * @param n_plans Number of plans
 * @param max_bytes Byte size to plan for arena
 * @param max_tensors Max tensors for arena
 * @return Planner* 
 */
Planner * planner_init(int n_plans, size_t max_bytes, int max_tensors);

/**
 * @brief Free planner 
 * 
 * @param planner 
 */
void free_planner(Planner * planner);

/**
 * @brief Add Plan in Planner. Mapping of Tensor name to index (arena location)
 * 
 * @param tensor_name Name of tensor 
 * @param idx Offset in arena to plan
 * @param planner Planner
 */
void planner_add(char * tensor_name, size_t idx,Planner * planner);

/**
 * @brief Get offset for tensor
 * 
 * @param tensor_name 
 * @param planner 
 * @return size_t 
 */
size_t get_plan(char * tensor_name, Planner * planner);

/**
 * @brief Construct Memory Planner from GraphProto. Allocations are done by First-Fit-Decreasing algorithm.
 * 
 * @param gproto GraphProto
 * @return Planner* 
 */
Planner * planner_init_from_proto(GraphProto * gproto);

#ifdef __cplusplus
}
#endif

#endif
