#include "uonnx.h"
#include "mnist_onnx.h"

// #define UONNX_ON_BREAKDOWN

Tensor *input, *output;
static float output_buf[10];
const char *labels[10] = {"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
int i = 0;

uint32_t ccount0 = 0;
uint32_t ccount1 = 0;

#ifndef UONNX_ON_BREAKDOWN
Context *ctx;
int mem0 = 0, mem1 = 0, mem2 = 0;
#else
ModelProto *model;
Graph *graph;
TensorArena *arena;
PlannerProto *planner;
Node *n;

int mem0 = 0, mem1 = 0, mem2 = 0, mem3 = 0, mem4 = 0, mem5 = 0;
#endif

static int argmax(float *buf, int len)
{
    if (len <= 0)
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

static void summary()
{
    Serial.print("Output: [ ");
    for (i = 0; i < 10; i++)
    {
        Serial.print(output_buf[i]);
        Serial.print(" ");
    }
    Serial.println("]");

    Serial.print("Pred: ");
    Serial.println(labels[argmax(output_buf, 10)]);

    Serial.print("uonnx_run cycles: ");
    Serial.println(ccount1 - ccount0);

#ifndef UONNX_ON_BREAKDOWN
    Serial.print("Runtime mem: ");
    Serial.println(mem2 - mem0);
#else
    Serial.print("Model mem: ");
    Serial.println(mem1 - mem0);
    Serial.print("Planner mem: ");
    Serial.println(mem2 - mem1);
    Serial.print("Arena mem: ");
    Serial.println(mem3 - mem2);
    Serial.print("Graph mem: ");
    Serial.println(mem4 - mem3);
    Serial.print("Total Runtime mem: ");
    Serial.println(mem5 - mem0);
#endif
    Serial.println("");
}

void setup()
{
    Serial.begin(115200);
    Serial.println("App starting...");

    ccount0 = rp2040.getCycleCount();
    delay(1000);
    ccount1 = rp2040.getCycleCount();

    Serial.print("Cycles per second: ");
    Serial.println(ccount1 - ccount0);

    mem0 = rp2040.getUsedHeap();

#ifndef UONNX_ON_BREAKDOWN
    ctx = uonnx_init(mnist_onnx, sizeof(mnist_onnx), mnist_planner, sizeof(mnist_planner));
    input = tensor_search(ctx->arena, "Input3");
    output = tensor_search(ctx->arena, "Plus214_Output_0");

#else
    model = load_model_buf(mnist_onnx, sizeof(mnist_onnx));
    mem1 = rp2040.getUsedHeap();
    planner = load_planner_buf(mnist_planner, sizeof(mnist_planner));
    mem2 = rp2040.getUsedHeap();
    arena = arena_init(planner->arena->max_ntensors, planner->arena->max_bytes);
    mem3 = rp2040.getUsedHeap();
    graph = graph_init_from_PlannerProto(model, planner, arena);
    mem4 = rp2040.getUsedHeap();

    input = tensor_search(arena, "Input3");
    output = tensor_search(arena, "Plus214_Output_0");
#endif
}

void loop()
{
    tensor_apply((void *)input_3, sizeof(input_3), input);

#ifndef UONNX_ON_BREAKDOWN
    ccount0 = rp2040.getCycleCount();
    uonnx_run(ctx);
    ccount1 = rp2040.getCycleCount();
    mem2 = rp2040.getUsedHeap();
#else
    ccount0 = rp2040.getCycleCount();
    for (i = 0; i < graph->nlen; i++)
    {
        n = &graph->nodes[i];
        n->op(n);
    }
    ccount1 = rp2040.getCycleCount();
    mem5 = rp2040.getUsedHeap();
#endif
    memcpy(output_buf, output->datas, sizeof(output_buf));
    summary();
}
