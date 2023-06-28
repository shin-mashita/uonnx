#include "uonnx.h"
#include "reference_onnx.h"
#include <malloc.h>

#define UONNX_ON_BREAKDOWN

volatile uint32_t *DWT_CONTROL = (uint32_t *)0xE0001000;
volatile uint32_t *DWT_CYCCNT = (uint32_t *)0xE0001004;
volatile uint32_t *DEMCR = (uint32_t *)0xE000EDFC;
volatile uint32_t *LAR = (uint32_t *)0xE0001FB0;

Tensor *input, *output;
static float output_buf[5];
const char *labels[5] = {"label1", "label2", "label3", "label4", "label5"};
int i = 0;

uint32_t ccount0 = 0;
uint32_t ccount1 = 0;

#ifndef UONNX_ON_BREAKDOWN
Context *ctx;
uint32_t mem0 = 0, mem1 = 0, mem2 = 0;
#else
ModelProto *model;
Graph *graph;
TensorArena *arena;
PlannerProto *planner;
Node *n;

uint32_t mem0 = 0, mem1 = 0, mem2 = 0, mem3 = 0, mem4 = 0, mem5 = 0;
#endif

static uint32_t get_heap()
{
    struct mallinfo mi = mallinfo();
    uint32_t heapRAM = (uint32_t)mi.uordblks;
    return heapRAM;
}

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
    for (i = 0; i < 5; i++)
    {
        Serial.print(output_buf[i]);
        Serial.print(" ");
    }
    Serial.println("]");

    Serial.print("Pred: ");
    Serial.println(labels[argmax(output_buf, 5)]);

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

    *DEMCR = *DEMCR | 0x01000000;    // enable trace
    *LAR = 0xC5ACCE55;               // <-- added unlock access to DWT (ITM, etc.)registers
    *DWT_CYCCNT = 0;                 // clear DWT cycle counter
    *DWT_CONTROL = *DWT_CONTROL | 1; // enable DWT cycle counter

    mem0 = get_heap();

#ifndef UONNX_ON_BREAKDOWN
    ctx = uonnx_init(reference_onnx, sizeof(reference_onnx), reference_planner_pb, sizeof(reference_planner_pb));
    input = tensor_search(ctx->arena, "serving_default_conv2d_15_input:0");
    output = tensor_search(ctx->arena, "StatefulPartitionedCall:0");

#else
    model = load_model_buf(reference_onnx, sizeof(reference_onnx));
    mem1 = get_heap();
    planner = load_planner_buf(reference_planner_pb, sizeof(reference_planner_pb));
    mem2 = get_heap();
    arena = arena_init(planner->arena->max_ntensors, planner->arena->max_bytes);
    mem3 = get_heap();
    graph = graph_init_from_PlannerProto(model, planner, arena);
    mem4 = get_heap();

    input = tensor_search(arena, "serving_default_conv2d_15_input:0");
    output = tensor_search(arena, "StatefulPartitionedCall:0");
#endif
}

void loop()
{
    tensor_apply((void *)ref_input, sizeof(ref_input), input);
    *DWT_CYCCNT = 0; // clear DWT cycle counter

#ifndef UONNX_ON_BREAKDOWN
    ccount0 = *DWT_CYCCNT;
    uonnx_run(ctx);
    ccount1 = *DWT_CYCCNT;
    mem2 = get_heap();
#else
    ccount0 = *DWT_CYCCNT;
    for (i = 0; i < graph->nlen; i++)
    {
        n = &graph->nodes[i];
        n->op(n);
    }
    ccount1 = *DWT_CYCCNT;
    mem5 = get_heap();
#endif
    memcpy(output_buf, output->datas, sizeof(output_buf));
    summary();
}
