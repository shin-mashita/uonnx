#include <uonnx.h>
#include "model.h"
// TODO: Tensor apply inconsistents with sizeof() and with actual indexes

int main()
{
    const char * filename = "./scratch/model.onnx";
    Context * ctx;
    Planner * planner;
    static float output_buf[10];

    // FFD per node
    planner = planner_init(13, sizeof(float)*(15204), 21);
    planner_add("Parameter193_reshape1", 0, planner); // fp32 (2560) - Node 0 10
    planner_add("Input3", sizeof(float)*(2560+6272), planner); // fp32 (784) - Node 1
    planner_add("Convolution28_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 1 2 
    planner_add("Plus30_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (6272) - Node 2 3 
    planner_add("ReLU32_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 3 4
    planner_add("Pooling66_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (1568) - Node 4 5
    planner_add("Convolution110_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 5 6
    planner_add("Plus112_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (3136) - Node 6 7
    planner_add("ReLU114_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 7 8
    planner_add("Pooling160_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (256) - Node 8 9 
    planner_add("Pooling160_Output_0_reshape0", sizeof(float)*(2560), planner); // fp32 (256) - 9 10
    planner_add("Times212_Output_0", sizeof(float)*(2560+256), planner); // fp32 (10) - Node 10 11
    planner_add("Plus214_Output_0", sizeof(float)*(0), planner); // fp32 (10) - Node 11

    ctx = uonnx_init(filename, NULL, 0, (void *)input_3, sizeof(input_3), "Input3", (void *)output_buf, sizeof(output_buf), "Plus214_Output_0", planner);

    int i = 1, j = 0;
    while(i--)
    {
        uonnx_run(ctx);
        for(j = 0; j < 10; j++)
            printf("%f ", output_buf[j]);
    }

    uonnx_free(ctx);

    return 0;
}

// int main()
// {
//     int i = 0, j = 0; 

//     ModelProto * model;
//     const char * filename = "./scratch/model.onnx";
//     model = load_model(filename);

//     // Graph * graph;
//     TensorArena * arena;
//     Planner * planner;
    
//     // FFD per node
//     planner = planner_init(13, sizeof(float)*(15204), 21);
//     planner_add("Parameter193_reshape1", 0, planner); // fp32 (2560) - Node 0 10
//     planner_add("Input3", sizeof(float)*(2560+6272), planner); // fp32 (784) - Node 1
//     planner_add("Convolution28_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 1 2 
//     planner_add("Plus30_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (6272) - Node 2 3 
//     planner_add("ReLU32_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 3 4
//     planner_add("Pooling66_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (1568) - Node 4 5
//     planner_add("Convolution110_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 5 6
//     planner_add("Plus112_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (3136) - Node 6 7
//     planner_add("ReLU114_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 7 8
//     planner_add("Pooling160_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (256) - Node 8 9 
//     planner_add("Pooling160_Output_0_reshape0", sizeof(float)*(2560), planner); // fp32 (256) - 9 10
//     planner_add("Times212_Output_0", sizeof(float)*(2560+256), planner); // fp32 (10) - Node 10 11
//     planner_add("Plus214_Output_0", sizeof(float)*(0), planner); // fp32 (10) - Node 11


    
//     arena = arena_init_from_planner(planner); // Initialization of arena

//     Graph * g = graph_init(model->graph, model, arena, planner);

//     const float test[] = {1,2,3,4,5,6,7,8,9,10};

//     Tensor * t = tensor_search(arena, "Input3");
//     tensor_apply((void*)input_3, sizeof(input_3), t);

//     // uonnx_run(ctx)
//     Node * n;
//     for(i = 0; i < g->nlen; i++)
//     {
//         n = &g->nodes[i];
//         n->operator(n);
//     }

//     // dump_arena(arena, TENSOR_TYPE_FLOAT32, 10);

//     dump_graph(g);
    
//     free_graph(g);
//     free_arena(arena);
//     free_planner(planner);
//     free_model(model);

//     return 0;
// }

    // FFD per node_io. DONT USE: Problems with overlapping inputs and outputs for MatMul.c
    // planner = planner_init(13, sizeof(float)*(2560+6272+100), 21);
    // planner_add("Parameter193_reshape1", 0, planner); // fp32 (2560) - Node 0 10
    // planner_add("Input3", sizeof(float)*(2560), planner); // fp32 (784) - Node 1
    // planner_add("Convolution28_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 1 2 
    // planner_add("Plus30_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 2 3 
    // planner_add("ReLU32_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 3 4
    // planner_add("Pooling66_Output_0", sizeof(float)*(2560), planner); // fp32 (1568) - Node 4 5
    // planner_add("Convolution110_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 5 6
    // planner_add("Plus112_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 6 7
    // planner_add("ReLU114_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 7 8
    // planner_add("Pooling160_Output_0", sizeof(float)*(2560), planner); // fp32 (256) - Node 8 9 
    // planner_add("Pooling160_Output_0_reshape0", sizeof(float)*(2560), planner); // fp32 (256) - 9 10
    // planner_add("Times212_Output_0", sizeof(float)*(0), planner); // fp32 (10) - Node 10 11
    // planner_add("Plus214_Output_0", sizeof(float)*(0), planner); // fp32 (10) - Node 11