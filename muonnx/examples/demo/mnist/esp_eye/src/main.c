/* Blink Example
   This example code is in the Public Domain (or CC0 licensed, at your option.)
   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sdkconfig.h"

#include <uonnx.h>
#include <model.h>
#include "esp_camera.h"

static const char *TAG = "example";
Context *ctx;
Planner *planner;

static float input_buf[784];
static float output_buf[10];
static int i = 0, j = 0, k = 0, max_idx = -1;
static float max_pred = 0;

#define BLINK_GPIO 2

// static uint8_t s_led_state = 0;
static uint32_t memfree_at_init = 0, memfree_at_run = 0, memfree_at_free = 0;
static uint32_t memuse_at_run = 0, memuse_at_free = 0;

// #define CAMERA_MODEL_AI_THINKER
// #define PWDN_GPIO_NUM     32
// #define RESET_GPIO_NUM    -1
// #define XCLK_GPIO_NUM      0
// #define SIOD_GPIO_NUM     26
// #define SIOC_GPIO_NUM     27
// #define Y9_GPIO_NUM       35
// #define Y8_GPIO_NUM       34
// #define Y7_GPIO_NUM       39
// #define Y6_GPIO_NUM       36
// #define Y5_GPIO_NUM       21
// #define Y4_GPIO_NUM       19
// #define Y3_GPIO_NUM       18
// #define Y2_GPIO_NUM        5
// #define VSYNC_GPIO_NUM    25
// #define HREF_GPIO_NUM     23
// #define PCLK_GPIO_NUM     22

#define CAMERA_MODEL_ESP_EYE
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5
#define Y2_GPIO_NUM 11
#define Y3_GPIO_NUM 9
#define Y4_GPIO_NUM 8
#define Y5_GPIO_NUM 10
#define Y6_GPIO_NUM 12
#define Y7_GPIO_NUM 18
#define Y8_GPIO_NUM 17
#define Y9_GPIO_NUM 16
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

camera_config_t config;
esp_err_t err;
camera_fb_t *fb;
int x_offset, y_offset;
uint8_t pixel;

void camera_config()
{
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.frame_size = FRAMESIZE_96X96;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count = 1;
}

void camera_init()
{
    err = esp_camera_init(&config);

    if (err != ESP_OK)
    {
        ESP_LOGI(TAG, "Camera init failed");
        return;
    }
    sensor_t *s = esp_camera_sensor_get();
    s->set_brightness(s, 0);
    s->set_contrast(s, 2);
    s->set_saturation(s, -2);
    s->set_exposure_ctrl(s, 0);
    s->set_ae_level(s, 2);
    s->set_aec_value(s, 1200);
    s->set_gain_ctrl(s, 0);
    s->set_agc_gain(s, 0);
    s->set_whitebal(s, 0);
    s->set_awb_gain(s, 0);
}

void get_frame()
{
    fb = esp_camera_fb_get();
    if (!fb)
    {
        ESP_LOGI(TAG, "Camera capture failed");
        return;
    }
    // crop the image to the center and skip every 3rd pixel
    x_offset = (fb->width - 28 * 3) / 2;
    y_offset = (fb->height - 28 * 3) / 2;

    // increase the contrast of the cropped image and send it to serial
    k = 0;
    for (i = y_offset; i < y_offset + 28 * 3; i += 3)
    {
        for (j = x_offset; j < x_offset + 28 * 3; j += 3)
        {
            // read the pixel value and apply contrast stretching
            pixel = fb->buf[i * fb->width + j];
            // pixel = map(pixel, 0, 255, 32, 223); // seems like this reduces contrast
            //  floor/ceil dark/light gray
            if (pixel < 64)
                pixel = 0;
            if (pixel >= 192)
                pixel = 255;
            input_buf[k] = (float)pixel;
            k++;
        }
    }

    ESP_LOGI(TAG, "%d", k);
    esp_camera_fb_return(fb);
}

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
        ESP_LOGI(TAG, "Pred: \"Zero\" | Mem Usage: %u B", memuse_at_run);
        break;
    case 1:
        ESP_LOGI(TAG, "Pred: \"One\" | Mem Usage: %u B", memuse_at_run);
        break;
    case 2:
        ESP_LOGI(TAG, "Pred: \"Two\" | Mem Usage: %u B", memuse_at_run);
        break;
    case 3:
        ESP_LOGI(TAG, "Pred: \"Three\" | Mem Usage: %u B", memuse_at_run);
        break;
    case 4:
        ESP_LOGI(TAG, "Pred: \"Four\" | Mem Usage: %u B", memuse_at_run);
        break;
    case 5:
        ESP_LOGI(TAG, "Pred: \"Five\" | Mem Usage: %u B", memuse_at_run);
        break;
    case 6:
        ESP_LOGI(TAG, "Pred: \"Six\" | Mem Usage: %u B", memuse_at_run);
        break;
    case 7:
        ESP_LOGI(TAG, "Pred: \"Seven\" | Mem Usage: %u B", memuse_at_run);
        break;
    case 8:
        ESP_LOGI(TAG, "Pred: \"Eight\" | Mem Usage: %u B", memuse_at_run);
        break;
    case 9:
        ESP_LOGI(TAG, "Pred: \"Nine\" | Mem Usage: %u B", memuse_at_run);
        break;
    default:
        ESP_LOGI(TAG, "Pred: \"Invalid\" | Mem Usage: %u B", memuse_at_run);
        break;
    }
}

void app_main(void)
{
    camera_config();
    camera_init();

    while(1)
    {
        get_frame();
    }
}

// void app_main(void)
// {
//     memfree_at_init = esp_get_free_heap_size();
//     // FFD per node
//     planner = planner_init(13, sizeof(float) * (15300), 21);
//     planner_add("Parameter193_reshape1", 0, planner);                             // fp32 (2560) - Node 0 10
//     planner_add("Input3", sizeof(float) * (2560 + 6272), planner);                // fp32 (784) - Node 1
//     planner_add("Convolution28_Output_0", sizeof(float) * (2560), planner);       // fp32 (6272) - Node 1 2
//     planner_add("Plus30_Output_0", sizeof(float) * (2560 + 6272), planner);       // fp32 (6272) - Node 2 3
//     planner_add("ReLU32_Output_0", sizeof(float) * (2560), planner);              // fp32 (6272) - Node 3 4
//     planner_add("Pooling66_Output_0", sizeof(float) * (2560 + 6272), planner);    // fp32 (1568) - Node 4 5
//     planner_add("Convolution110_Output_0", sizeof(float) * (2560), planner);      // fp32 (3136) - Node 5 6
//     planner_add("Plus112_Output_0", sizeof(float) * (2560 + 3136), planner);      // fp32 (3136) - Node 6 7
//     planner_add("ReLU114_Output_0", sizeof(float) * (2560), planner);             // fp32 (3136) - Node 7 8
//     planner_add("Pooling160_Output_0", sizeof(float) * (2560 + 3136), planner);   // fp32 (256) - Node 8 9
//     planner_add("Pooling160_Output_0_reshape0", sizeof(float) * (2560), planner); // fp32 (256) - 9 10
//     planner_add("Times212_Output_0", sizeof(float) * (2560 + 256), planner);      // fp32 (10) - Node 10 11
//     planner_add("Plus214_Output_0", sizeof(float) * (0), planner);                // fp32 (10) - Node 11

//     ctx = uonnx_init(NULL, mnist_onnx, sizeof(mnist_onnx), (void *)input_3, sizeof(input_3), "Input3", (void *)output_buf, sizeof(output_buf), "Plus214_Output_0", planner);

//     while (1)
//     {
//         uonnx_run(ctx);
//         memfree_at_run = esp_get_free_heap_size();
//         memuse_at_run = memfree_at_init - memfree_at_run;
//         output_argmax();
//         print_label();
//     }

//     uonnx_free(ctx);

//     memfree_at_free = esp_get_free_heap_size();
//     memuse_at_free = memfree_at_init - memfree_at_free;
// }

