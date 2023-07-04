#include "uonnx.h"
#include "model.h"
#include <malloc.h>
// TODO: Tensor apply inconsistents with sizeof() and with actual indexes

int main()
{
    const char * filename = "./scratch/model.onnx";
    ModelProto * model;
    Context * ctx;
    Planner * planner;
    static float output_buf[10];

    // FFD per node

    model = load_model(filename);
    planner = planner_init_from_proto(model->graph);
    dump_planner(planner);

    // planner = planner_init(13, sizeof(float)*(15204), 21);
    // planner_add("Parameter193_reshape1", 0, planner); // fp32 (2560) - Node 0 10
    // planner_add("Input3", sizeof(float)*(2560+6272), planner); // fp32 (784) - Node 1
    // planner_add("Convolution28_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 1 2 
    // planner_add("Plus30_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (6272) - Node 2 3 
    // planner_add("ReLU32_Output_0", sizeof(float)*(2560), planner); // fp32 (6272) - Node 3 4
    // planner_add("Pooling66_Output_0", sizeof(float)*(2560+6272), planner); // fp32 (1568) - Node 4 5
    // planner_add("Convolution110_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 5 6
    // planner_add("Plus112_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (3136) - Node 6 7
    // planner_add("ReLU114_Output_0", sizeof(float)*(2560), planner); // fp32 (3136) - Node 7 8
    // planner_add("Pooling160_Output_0", sizeof(float)*(2560+3136), planner); // fp32 (256) - Node 8 9 
    // planner_add("Pooling160_Output_0_reshape0", sizeof(float)*(2560), planner); // fp32 (256) - 9 10
    // planner_add("Times212_Output_0", sizeof(float)*(2560+256), planner); // fp32 (10) - Node 10 11
    // planner_add("Plus214_Output_0", sizeof(float)*(0), planner); // fp32 (10) - Node 11

    // dump_planner(planner);
    // planner = planner_init(13, sizeof(float)*(33668), 21);
    // planner_add("Parameter193_reshape1", 0, planner); // fp32 (2560) - Node 0 10
    // planner_add("Input3", sizeof(float)*(2560), planner); // fp32 (784) - Node 1
    // planner_add("Convolution28_Output_0", sizeof(float)*(3344), planner); // fp32 (6272) - Node 1 2 
    // planner_add("Plus30_Output_0", sizeof(float)*(9616), planner); // fp32 (6272) - Node 2 3 
    // planner_add("ReLU32_Output_0", sizeof(float)*(15888), planner); // fp32 (6272) - Node 3 4
    // planner_add("Pooling66_Output_0", sizeof(float)*(22160), planner); // fp32 (1568) - Node 4 5
    // planner_add("Convolution110_Output_0", sizeof(float)*(23728), planner); // fp32 (3136) - Node 5 6
    // planner_add("Plus112_Output_0", sizeof(float)*(26864), planner); // fp32 (3136) - Node 6 7
    // planner_add("ReLU114_Output_0", sizeof(float)*(30000), planner); // fp32 (3136) - Node 7 8
    // planner_add("Pooling160_Output_0", sizeof(float)*(33136), planner); // fp32 (256) - Node 8 9 
    // planner_add("Pooling160_Output_0_reshape0", sizeof(float)*(33392), planner); // fp32 (256) - 9 10
    // planner_add("Times212_Output_0", sizeof(float)*(33648), planner); // fp32 (10) - Node 10 11
    // planner_add("Plus214_Output_0", sizeof(float)*(33658), planner); // fp32 (10) - Node 11

    ctx = uonnx_init(NULL, mnist_onnx, sizeof(mnist_onnx), (void *)input_3, sizeof(input_3), "Input3", (void *)output_buf, sizeof(output_buf), "Plus214_Output_0", planner);

    int i = 1, j = 0;
    while(i--)
    {
        uonnx_run(ctx);
        for(j = 0; j < 10; j++)
            printf("%f ", output_buf[j]);
    }

    printf("\n");

    uonnx_free(ctx);

    return 0;
}

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