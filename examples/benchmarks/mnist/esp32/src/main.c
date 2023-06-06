#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sdkconfig.h"

#include <uonnx.h>
#include <model.h>

static const char *TAG = "example";
Context *ctx;
Planner *planner;

// ModelProto * model;
// Graph * graph;
// TensorArena * arena;

static float output_buf[10];
static int i = 0, j = 0, k = 0, max_idx = -1;
static float max_pred = 0;

static uint32_t memfree_at_init = 0, memfree_at_run = 0, memfree_at_free = 0, memfreemin_at_run=0, memfreecur =  0;
static uint32_t memuse_at_run = 0, memuse_at_free = 0, memusemax_at_run = 0;

esp_cpu_ccount_t count0 = 0, count1 =0; 

void output_argmax()
{
    max_idx = -1;
    max_pred = output_buf[0];

    for (j = 0; j < 10; j++)
    {
        if (output_buf[j] >= max_pred)
        {
            max_pred = output_buf[j];
            max_idx = j;
        }
    }
}

void print_label()
{
    switch (max_idx)
    {
    case 0:
        ESP_LOGI(TAG, "Pred: \"Zero\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    case 1:
        ESP_LOGI(TAG, "Pred: \"One\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    case 2:
        ESP_LOGI(TAG, "Pred: \"Two\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    case 3:
        ESP_LOGI(TAG, "Pred: \"Three\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    case 4:
        ESP_LOGI(TAG, "Pred: \"Four\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    case 5:
        ESP_LOGI(TAG, "Pred: \"Five\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    case 6:
        ESP_LOGI(TAG, "Pred: \"Six\" | Mem Usage: %u B| Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    case 7:
        ESP_LOGI(TAG, "Pred: \"Seven\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    case 8:
        ESP_LOGI(TAG, "Pred: \"Eight\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    case 9:
        ESP_LOGI(TAG, "Pred: \"Nine\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    default:
        ESP_LOGI(TAG, "Pred: \"Invalid\" | Mem Usage: %u B | Max Mem Usage: %u B", memuse_at_run, memusemax_at_run);
        break;
    }
}

void log_current_mem(char * mtag)
{
    memfreecur = esp_get_free_heap_size();
    ESP_LOGI(mtag, "Current memory usage is: %u", memfree_at_init - memfreecur);
}

void app_main(void)
{
    ESP_LOGI(TAG, "App starting...");
    memfree_at_init = esp_get_free_heap_size();

    planner = planner_init(13, sizeof(float) * (15300), 21);
    log_current_mem("planner");
    planner_add("Parameter193_reshape1", 0, planner);                             // fp32 (2560) - Node 0 10
    planner_add("Input3", sizeof(float) * (2560 + 6272), planner);                // fp32 (784) - Node 1
    planner_add("Convolution28_Output_0", sizeof(float) * (2560), planner);       // fp32 (6272) - Node 1 2
    planner_add("Plus30_Output_0", sizeof(float) * (2560 + 6272), planner);       // fp32 (6272) - Node 2 3
    planner_add("ReLU32_Output_0", sizeof(float) * (2560), planner);              // fp32 (6272) - Node 3 4
    planner_add("Pooling66_Output_0", sizeof(float) * (2560 + 6272), planner);    // fp32 (1568) - Node 4 5
    planner_add("Convolution110_Output_0", sizeof(float) * (2560), planner);      // fp32 (3136) - Node 5 6
    planner_add("Plus112_Output_0", sizeof(float) * (2560 + 3136), planner);      // fp32 (3136) - Node 6 7
    planner_add("ReLU114_Output_0", sizeof(float) * (2560), planner);             // fp32 (3136) - Node 7 8
    planner_add("Pooling160_Output_0", sizeof(float) * (2560 + 3136), planner);   // fp32 (256) - Node 8 9
    planner_add("Pooling160_Output_0_reshape0", sizeof(float) * (2560), planner); // fp32 (256) - 9 10
    planner_add("Times212_Output_0", sizeof(float) * (2560 + 256), planner);      // fp32 (10) - Node 10 11
    planner_add("Plus214_Output_0", sizeof(float) * (0), planner);                // fp32 (10) - Node 11

    ctx = uonnx_init(NULL, mnist_onnx, sizeof(mnist_onnx), (void *)input_3, sizeof(input_3), "Input3", (void *)output_buf, sizeof(output_buf), "Plus214_Output_0", planner);
    
    // model = load_model_buf(mnist_onnx, sizeof(mnist_onnx));
    // log_current_mem("model");
    // arena = arena_init_from_planner(planner);
    // log_current_mem("arena");
    // graph = graph_init(model->graph, model, arena, planner);
    // log_current_mem("graph");

    while(1)
    {
        count0 = esp_cpu_get_ccount();
        uonnx_run(ctx);
        count1 = esp_cpu_get_ccount();
        ESP_LOGI(TAG, "%u",count0);
        ESP_LOGI(TAG, "%u",count1);
        ESP_LOGI(TAG, "%u",count1 - count0);
        memfree_at_run = esp_get_free_heap_size();
        memfreemin_at_run = esp_get_minimum_free_heap_size();
        memuse_at_run = memfree_at_init - memfree_at_run;    
        memusemax_at_run = memfree_at_init - memfreemin_at_run; 

        output_argmax();
        print_label();
    }

    uonnx_free(ctx);

    // free_graph(graph);
    // free_arena(arena);
    // free_model(model);
    // free_planner(planner);

    memfree_at_free = esp_get_free_heap_size();
    memuse_at_free = memfree_at_init - memfree_at_free;
}
