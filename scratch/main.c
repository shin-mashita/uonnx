#include "uonnx.h"
// #include "model.h"
#include "uonnx_mnist.h"
#include <malloc.h>
// TODO: Tensor apply inconsistents with sizeof() and with actual indexes

int main()
{
    Context * ctx = uonnx_init(mnist_onnx, sizeof(mnist_onnx), mnist_planner, sizeof(mnist_planner));

    Tensor * input = tensor_search(ctx->arena, "Input3");
    Tensor * output = tensor_search(ctx->arena, "Plus214_Output_0");
    tensor_apply((void *)input_3, sizeof(input_3), input);

    uonnx_run(ctx);

    dump_graph(ctx->graph);
    uonnx_free(ctx);
    
    return 0;
}