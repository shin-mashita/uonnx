#include <uonnx.h>

Tensor * tensor_init(const char *name, TensorType type, int *dims, int ndim, TensorArena *arena, Tensor * tensor, int data_idx);
Tensor * tensor_init_from_value_info(ValueInfoProto * v, TensorArena * arena, Tensor * tensor, int data_idx);
Tensor * tensor_init_from_proto(TensorProto * tp, Tensor * tensor);

void free_tensor(Tensor * t);
void free_tensor_from_arena(Tensor * t);
void tensor_apply(void * datas, size_t size, Tensor * t);

Graph * graph_init(GraphProto * gproto, ModelProto * model, TensorArena * arena, Planner * planner); // WIP
void free_graph(Graph * g); 