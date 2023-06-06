#ifndef __UONNX_PLANNER_H__
#define __UONNX_PLANNER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <uonnx.h>

Plan * plan_create(char * tensor_name, size_t idx);
void free_plan(Plan * p);
Planner * planner_init(int n_plans, size_t max_bytes, int max_tensors);
void free_planner(Planner * planner);
void planner_add(char * tensor_name, size_t idx,Planner * planner);
size_t get_plan(char * tensor_name, Planner * planner);

// Planner * planner_init_from_GraphProto(GraphProto * gproto); //TODO

#ifdef __cplusplus
}
#endif

#endif
