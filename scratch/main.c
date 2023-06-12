#include <uonnx.h>
#include "model.h"
#include <malloc.h>
// TODO: Tensor apply inconsistents with sizeof() and with actual indexes

printMemoryInfo()
{
   struct mallinfo mi;
   memset(&mi,0,sizeof(struct mallinfo));
   mi = mallinfo();
   printf("Current Usage: %d\n",mi.uordblks);
}

void read_planner(PlannerProto * planner)
{
    int i = 0, j = 0;

    printf("Arena max tensors: %d\n", planner->arena->max_ntensors);
    printf("Arena max bytes: %d\n", planner->arena->max_bytes);
    
    for(i = 0; i < planner->n_plans; i++)
    {
        printf("Planner: %s\n", planner->plans[i]->name);
        printf("\tstart_idx: %d\n", planner->plans[i]->start_idx);
        printf("\tsize: %d\n", planner->plans[i]->size);
        printf("\tndata: %d\n", planner->plans[i]->ndata);
        printf("\tndims: %d\n", planner->plans[i]->n_dims);
        printf("\tdims: ");
        for(j = 0; j < planner->plans[i]->n_dims; j++)
        {
            printf("%d ", planner->plans[i]->dims[j]);
        }
        printf("\n");
    }
}

int main()
{
    ModelProto * model;
    PlannerProto * planner;
    TensorArena * arena;
    Graph * g;
    
    printMemoryInfo();
    model = load_model("./scratch/model.onnx");
    // model = load_model("./scratch/kws_float32_9.onnx");
    printMemoryInfo();
    planner = load_planner("./scratch/model_planner.pb");
    // planner = load_planner("./scratch/kws_float32_9_planner.pb");
    printMemoryInfo();
    
    
    arena = arena_init_v2(planner->arena->max_ntensors + model->graph->n_initializer, planner->arena->max_bytes);

    // read_planner(planner);
    
    g = graph_init_from_PlannerProto(model->graph, model, planner, arena);

    // Tensor * t = tensor_search(arena, "input_1");
    // tensor_apply((void*)(input_1), sizeof(input_1), t);

    Tensor * t = tensor_search(arena, "Input3");
    tensor_apply((void*)(input_3), sizeof(input_3), t);

    printMemoryInfo();
    Node *n;

    for(int i = 0; i < g->nlen; i++)
    {
        n = &g->nodes[i];
        n->operator(n);
    }

    printMemoryInfo();
    // dump_graph(g);

    free_graph(g);
    free_arena(arena);
    free_plannerproto(planner);
    free_model(model);
    printMemoryInfo();
    
    return 0;
}

// int main()
// {
//     const char * filename = "./scratch/test.onnx";
//     ModelProto * model;
//     Context * ctx;
//     Planner * planner;
//     static float output_buf[12];

//     // planner = load_planner("./scratch/planner.pb");

//     // read_planner(planner);
//     // FFD per node

//     model = load_model(filename);
//     planner = planner_init_from_proto(model->graph);
//     printf("%d\n", planner->max_arena_size);
//     dump_planner(planner);

//     // planner = planner_init(13, sizeof(float)*(15204), 21);
//     // planner_add("Parameter193_reshape1", 0, planner); // fp32 (2560) - Node 0 10
//     // planner_add("Input3", sizeof(float)*(2560+6272), planner); // fp32 (784) - Node 1
//     // planner_add("Convolution28_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 1 2 
//     // planner_add("Plus30_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (6272) - Node 2 3 
//     // planner_add("ReLU32_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 3 4
//     // planner_add("Pooling66_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (1568) - Node 4 5
//     // planner_add("Convolution110_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 5 6
//     // planner_add("Plus112_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (3136) - Node 6 7
//     // planner_add("ReLU114_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 7 8
//     // planner_add("Pooling160_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (256) - Node 8 9 
//     // planner_add("Pooling160_Output_0_reshape0", sizeof(float)*(2560), planner); // fp32 (256) - 9 10
//     // planner_add("Times212_Output_0", sizeof(float)*(2560+256), planner); // fp32 (10) - Node 10 11
//     // planner_add("Plus214_Output_0", sizeof(float)*(0), planner); // fp32 (10) - Node 11

//     // dump_planner(planner);
//     // planner = planner_init(13, sizeof(float)*(33668), 21);
//     // planner_add("Parameter193_reshape1", 0, planner); // fp32 (2560) - Node 0 10
//     // planner_add("Input3", sizeof(float)*(2560), planner); // fp32 (784) - Node 1
//     // planner_add("Convolution28_Output_0", sizeof(float)*(3344), planner); // fp32 (6272) - Node 1 2 
//     // planner_add("Plus30_Output_0", sizeof(float)*(9616), planner); // fp32 (6272) - Node 2 3 
//     // planner_add("ReLU32_Output_0", sizeof(float)*(15888), planner); // fp32 (6272) - Node 3 4
//     // planner_add("Pooling66_Output_0", sizeof(float)*(22160), planner); // fp32 (1568) - Node 4 5
//     // planner_add("Convolution110_Output_0", sizeof(float)*(23728), planner); // fp32 (3136) - Node 5 6
//     // planner_add("Plus112_Output_0", sizeof(float)*(26864), planner); // fp32 (3136) - Node 6 7
//     // planner_add("ReLU114_Output_0", sizeof(float)*(30000), planner); // fp32 (3136) - Node 7 8
//     // planner_add("Pooling160_Output_0", sizeof(float)*(33136), planner); // fp32 (256) - Node 8 9 
//     // planner_add("Pooling160_Output_0_reshape0", sizeof(float)*(33392), planner); // fp32 (256) - 9 10
//     // planner_add("Times212_Output_0", sizeof(float)*(33648), planner); // fp32 (10) - Node 10 11
//     // planner_add("Plus214_Output_0", sizeof(float)*(33658), planner); // fp32 (10) - Node 11

//     ctx = uonnx_init("./scratch/test.onnx", NULL, 0, (void *)input_1, sizeof(input_1), "input_1", (void *)output_buf, sizeof(output_buf), "Identity", planner);

//     // int i = 1, j = 0;
//     // while(i--)
//     // {
//     //     uonnx_run(ctx);
//     //     for(j = 0; j < 12; j++)
//     //         printf("%f ", output_buf[j]);
//     // }

//     // printf("\n");

//     // uonnx_free(ctx);

//     return 0;
// }