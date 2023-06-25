#include <uonnx.h>
#include "kws_float32_9_onnx.h"

// #define UONNX_ON_BREAKDOWN

static int get_current_mem()
{
   struct mallinfo info;
   memset(&info,0,sizeof(struct mallinfo));
   info = mallinfo();
   return (info.uordblks + info.hblkhd);
}

static int argmax(float * buf, int len)
{
    if(len <= 0)
        return -1;

    int j = 0;
    int max_idx = -1;
    float max_pred = buf[0];

    for (j = 0; j < len; j++)
    {
        if (buf[j] >= max_pred)
        {
            max_pred = buf[j];
            max_idx = j;
        }
    }

    return max_idx;
}

int main()
{
    const char * labels[12] = {"go", "left", "no", "off", "on", "right", "stop", "up", "yes", "silence", "unknown"};
    float output[12];

    #ifndef UONNX_ON_BREAKDOWN

    int i = 0;
    int mem0, mem1;
    Tensor * input_T, * output_T;
    Context * ctx;

    mem0 = get_current_mem();
    ctx = uonnx_init(kws_float32_9_onnx, sizeof(kws_float32_9_onnx), kws_float32_9_planner, sizeof(kws_float32_9_planner));

    input_T = tensor_search(ctx->arena, "input_1");
    output_T = tensor_search(ctx->arena, "Identity");
    tensor_apply((void *)input_1, sizeof(input_1), input_T);

    uonnx_run(ctx);
    mem1 = get_current_mem();

    memcpy(output, output_T->datas, sizeof(output));
    uonnx_free(ctx);

    printf("\n\nOutput: [ ");
    for(i = 0; i < 12; i++)
        printf("%f ", output[i]);
    printf("]\n");
    printf("Pred: %s\n", labels[argmax(output, 12)]);
    printf("Memory Usage Summary:\n");
    printf("\tOn RUNTIME (B): %d\n", mem1-mem0);


    #else

    int i = 0;
    int mem0, mem1, mem2, mem3, mem4, mem5;
    ModelProto * model;
    PlannerProto * planner;
    TensorArena * arena;
    Graph * graph;
    Tensor * input_T, * output_T;
    Node * n;

    mem0 = get_current_mem();

    model = load_model_buf(kws_float32_9_onnx, sizeof(kws_float32_9_onnx));
    mem1 = get_current_mem();
    planner = load_planner_buf(kws_float32_9_planner, sizeof(kws_float32_9_planner));
    mem2 = get_current_mem();
    arena = arena_init(planner->arena->max_ntensors, planner->arena->max_bytes);
    mem3 = get_current_mem();
    graph = graph_init_from_PlannerProto(model, planner, arena);
    mem4 = get_current_mem();

    input_T = tensor_search(arena, "input_1");
    output_T = tensor_search(arena, "Identity");
    tensor_apply((void *)input_1, sizeof(input_1), input_T);

    for(i = 0; i < graph->nlen; i++)
    {
        n = &graph->nodes[i];
        n->op(n);
    }
    mem5 = get_current_mem();

    memcpy(output, output_T->datas, sizeof(output));

    free_graph(graph);
    free_arena(arena);
    free_model(model);
    free_plannerproto(planner);

    printf("\n\nOutput: [ ");
    for(i = 0; i < 12; i++)
        printf("%f ", output[i]);
    printf("]\n");
    printf("Pred: %s\n", labels[argmax(output, 12)]);
    printf("Memory Usage:\n");
    printf("\tOn MODEL (B): %d\n", mem1-mem0);
    printf("\tOn PLANNER (B): %d\n", mem2-mem1);
    printf("\tOn ARENA (B): %d\n", mem3-mem2);
    printf("\tOn GRAPH (B): %d\n", mem4-mem3);
    printf("\tOn RUNTIME (B): %d\n", mem5-mem0);

    #endif

    return 0;
}
