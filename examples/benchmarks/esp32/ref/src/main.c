#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sdkconfig.h"

#include "uonnx.h"
#include "reference_onnx.h"

#define UONNX_ON_BREAKDOWN

static const char *TAG = "REF_ESP32";

#ifndef UONNX_ON_BREAKDOWN
Context *ctx;
#else
ModelProto * model;
Graph * graph;
TensorArena * arena;
PlannerProto * planner;
Node * n;
int i = 0;
#endif

static float output_buf[5];
int max_idx = -1;
int j = 0;
float max_pred = -9999999;

Tensor * input, * output;

static uint32_t memfree_init = 0; 
static uint32_t memfree_free = 0;
static uint32_t memfree_now =  0;

#ifdef UONNX_ON_BREAKDOWN
static uint32_t memfree_ref =  0;
#endif

static uint32_t memuse_at_run = 0; 
static uint32_t memusemax_at_run = 0;

esp_cpu_ccount_t count0 = 0, count1 =0; 

void output_argmax()
{
    j = 0;
    max_idx = -1;
    max_pred = output_buf[0];

    for (j = 0; j < 5; j++)
    {
        if (output_buf[j] >= max_pred)
        {
            max_pred = output_buf[j];
            max_idx = j;
        }
    }
}

void print_output()
{
    switch(max_idx)
    {
    case 0:
        ESP_LOGI(TAG, "Pred: \"Label1\" | Mem Usage: %u B | Max Mem Usage: %u B | Cycles: %u", memuse_at_run, memusemax_at_run, count1- count0);
        break;
    case 1:
        ESP_LOGI(TAG, "Pred: \"Label2\" | Mem Usage: %u B | Max Mem Usage: %u B | Cycles: %u", memuse_at_run, memusemax_at_run, count1- count0);
        break;
    case 2:
        ESP_LOGI(TAG, "Pred: \"Label3\" | Mem Usage: %u B | Max Mem Usage: %u B | Cycles: %u", memuse_at_run, memusemax_at_run, count1- count0);
        break;
    case 3:
        ESP_LOGI(TAG, "Pred: \"Label4\" | Mem Usage: %u B | Max Mem Usage: %u B | Cycles: %u", memuse_at_run, memusemax_at_run, count1- count0);
        break;
    case 4:
        ESP_LOGI(TAG, "Pred: \"Label5\" | Mem Usage: %u B | Max Mem Usage: %u B | Cycles: %u", memuse_at_run, memusemax_at_run, count1- count0);
        break;
    default:
        ESP_LOGI(TAG, "Pred: \"Invalid\" | Mem Usage: %u B | Max Mem Usage: %u B | Cycles: %u", memuse_at_run, memusemax_at_run, count1- count0);
        break;
    }
}

void get_current_mem(const char * mtag, uint32_t memref)
{
    memfree_now = esp_get_free_heap_size();
    ESP_LOGI(mtag, "Current memory usage is: %u", memref - memfree_now);
}

void app_main(void)
{
    ESP_LOGI(TAG, "App starting...");
    memfree_init = esp_get_free_heap_size();
    get_current_mem("App init", memfree_init);

    #ifndef UONNX_ON_BREAKDOWN

    ctx = uonnx_init(reference_onnx, sizeof(reference_onnx), reference_planner_pb, sizeof(reference_planner_pb));
    get_current_mem("Ctx init", memfree_init);

    input = tensor_search(ctx->arena, "serving_default_conv2d_15_input:0");
    output = tensor_search(ctx->arena, "StatefulPartitionedCall:0");

    #else 

    memfree_ref = esp_get_free_heap_size();
    model = load_model_buf(reference_onnx, sizeof(reference_onnx));
    get_current_mem("model", memfree_ref);

    memfree_ref = esp_get_free_heap_size();
    planner = load_planner_buf(reference_planner_pb, sizeof(reference_planner_pb));
    get_current_mem("planner", memfree_ref);

    memfree_ref = esp_get_free_heap_size();
    arena = arena_init(planner->arena->max_ntensors, planner->arena->max_bytes);
    get_current_mem("arena", memfree_ref);

    memfree_ref = esp_get_free_heap_size();
    graph = graph_init_from_PlannerProto(model, planner, arena);
    get_current_mem("graph", memfree_ref);

    input = tensor_search(arena, "serving_default_conv2d_15_input:0");
    output = tensor_search(arena, "StatefulPartitionedCall:0");

    #endif

    while(1)
    {
        tensor_apply((void *)ref_input, sizeof(ref_input), input);
        count0 = esp_cpu_get_ccount();

        #ifndef UONNX_ON_BREAKDOWN
        uonnx_run(ctx);
        #else
        for(i = 0; i < graph->nlen; i++)
        {
            n = &graph->nodes[i];
            n->op(n);
        }
        #endif

        memuse_at_run = memfree_init - esp_get_free_heap_size();
        memusemax_at_run = memfree_init - esp_get_minimum_free_heap_size();

        count1 = esp_cpu_get_ccount();

        memcpy(output_buf, output->datas, sizeof(output_buf));
        output_argmax();
        print_output();
    }

    #ifndef UONNX_ON_BREAKDOWN
    uonnx_free(ctx);
    #else
    free_graph(graph);
    free_arena(arena);
    free_model(model);
    free_plannerproto(planner);
    #endif

    memfree_free = esp_get_free_heap_size();
}
