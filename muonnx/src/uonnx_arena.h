#include <uonnx.h>

TensorArena * arena_init(const int MAX_TENSORS, const size_t MAX_BYTES);
TensorArena * arena_init_from_planner(Planner * planner);
void free_arena(TensorArena * arena);
Tensor * tensor_search(TensorArena * arena, char * name);