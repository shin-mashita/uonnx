#include <uonnx.h>
#include "model.h"
// TODO: Tensor apply inconsistents with sizeof() and with actual indexes

int main()
{
    int i = 0, j = 0; 

    ModelProto * model = NULL;
    GraphProto * graph = NULL;
    Tensor * tv0;

    const char * filename = "./scratch/model.onnx";
    
    model = load_model(filename);

    ValueInfoProto * vp0 = model->graph->input[0];
    TensorProto * tp0 = model->graph->initializer[0];

    Planner * planner = planner_init(13, sizeof(float)*(2560+6272+6272+100), 21);
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


    TensorArena * arena = arena_init_from_planner(planner); // Initialization of arena


    // NEEDED: Convert int64_t dims first into int dims
    // int * t0dims = malloc(sizeof(int)*tp0->n_dims);
    // for(i = 0; i < tp0->n_dims; i++)
    // {
    //     t0dims[i] = tp0->dims[i];
    // }

    // Tensor * t0 = tensor_init(tp0->name, (TensorType)tp0->data_type, t0dims, tp0->n_dims, arena, 0, 1*sizeof(float));
    // dump_tensor(t0);

    // free(t0dims); // NEEDED

    // Tensor * t1 = tensor_init_from_value_info(vp0, arena, 9, 4*sizeof(float));

    // static const float testdata0[] = {1,2,3,4,5};
    // static const float testdata1[] = {6,7,8,9,10};

    // Tensor * test = tensor_search(arena, "Input3");
    // tensor_apply(testdata0, sizeof(testdata0), test);

    // test = tensor_search(arena, "Parameter193");
    // tensor_apply(testdata1, sizeof(testdata1), test);

    printf("%x\n", shash("Test_Op"));//0x39d6d0e3

    Graph * g = graph_init(model->graph, model, arena, planner);
    Node * n;

    const float test[] = {1,2,3,4,5,6,7,8,9,10};

    Tensor * t = tensor_search(arena, "Input3");
    tensor_apply((void*)input_3, sizeof(input_3), t);

    // uonnx_run(ctx)
    for(i = 0; i < g->nlen; i++)
    {
        n = &g->nodes[i];
        n->operator(n);
    }
    // dump_node(test_node);
    // printf("%d\n",test_node->opset);
    // test_node->operator(test_node);
    // dump_node(test_node);

    dump_graph(g);


    // printf("%d\n", arena->n_bytes);
    // dump_planner(planner);
    free_planner(planner);


    // dump_arena(arena, TENSOR_TYPE_FLOAT32, 100);

    // for(i = 0; i < model->graph->n_input; i++)
    // {
    //     printf("%s\n", model->graph->input[i]->name);
    // }
    // printf("\n\n");
    // for(i = 0; i < model->graph->n_output; i++)
    // {
    //     printf("%s\n", model->graph->output[i]->name);
    // }
    // printf("\n\n");
    // for(i = 0; i < model->graph->n_value_info; i++)
    // {
    //     printf("%s\n", model->graph->value_info[i]->name);
    // }
    // printf("\n\n");
    // for(i = 0; i < model->graph->n_initializer; i++)
    // {
    //     printf("%s\n", model->graph->initializer[i]->name);
    // }


    return 0;
}
