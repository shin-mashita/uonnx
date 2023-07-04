#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "sdkconfig.h"

#include "uonnx.h"
#include <mnist_onnx.h>
#include "esp_camera.h"

static const char *TAG = "DEMO";
Context *ctx;
Tensor * input, * output;

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

#if defined(CAMERA_MODEL_ESP_EYE)

#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    4
#define SIOD_GPIO_NUM    18
#define SIOC_GPIO_NUM    23

#define Y9_GPIO_NUM      36
#define Y8_GPIO_NUM      37
#define Y7_GPIO_NUM      38
#define Y6_GPIO_NUM      39
#define Y5_GPIO_NUM      35
#define Y4_GPIO_NUM      14
#define Y3_GPIO_NUM      13
#define Y2_GPIO_NUM      34
#define VSYNC_GPIO_NUM   5
#define HREF_GPIO_NUM    27
#define PCLK_GPIO_NUM    25

#endif

camera_config_t config;
esp_err_t err;
camera_fb_t *fb;
int x_offset, y_offset;
uint8_t pixel;

void camera_config()
{
    
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
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
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    // config.pin_d0 = Y2_GPIO_NUM;
    // config.pin_d1 = Y3_GPIO_NUM;
    // config.pin_d2 = Y4_GPIO_NUM;
    // config.pin_d3 = Y5_GPIO_NUM;
    // config.pin_d4 = Y6_GPIO_NUM;
    // config.pin_d5 = Y7_GPIO_NUM;
    // config.pin_d6 = Y8_GPIO_NUM;
    // config.pin_d7 = Y9_GPIO_NUM;
    // config.pin_xclk = XCLK_GPIO_NUM;
    // config.pin_pclk = PCLK_GPIO_NUM;
    // config.pin_vsync = VSYNC_GPIO_NUM;
    // config.pin_href = HREF_GPIO_NUM;
    // config.pin_sscb_sda = SIOD_GPIO_NUM;
    // config.pin_sscb_scl = SIOC_GPIO_NUM;
    // config.pin_pwdn = PWDN_GPIO_NUM;
    // config.pin_reset = RESET_GPIO_NUM;
    // config.xclk_freq_hz = 20000000;
    // config.frame_size = FRAMESIZE_96X96;
    // config.pixel_format = PIXFORMAT_GRAYSCALE;
    // config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    // config.fb_location = CAMERA_FB_IN_PSRAM;
    // config.fb_count = 1;

    #if defined(CAMERA_MODEL_ESP_EYE)
        err = gpio_set_direction(13, GPIO_MODE_INPUT);
        if(err != ESP_OK) ESP_LOGE(TAG, "Failed to set gpio 13 input");
        err = gpio_set_direction(14, GPIO_MODE_INPUT);
        if(err != ESP_OK) ESP_LOGE(TAG, "Failed to set gpio 14 input");

        gpio_set_pull_mode(13, GPIO_PULLUP_ONLY);
        if(err != ESP_OK) ESP_LOGE(TAG, "Failed to set gpio 13 pullup");
        gpio_set_pull_mode(14, GPIO_PULLUP_ONLY);
        if(err != ESP_OK) ESP_LOGE(TAG, "Failed to set gpio 14 pullup");
    #endif
}

void camera_init()
{
    err = esp_camera_init(&config);

    if (err != ESP_OK)
    {
        while(1)
        {
            ESP_LOGE(TAG, "Camera init failed with err 0x%x", err);
        }
    }
    sensor_t *s = esp_camera_sensor_get();
    // s->set_brightness(s, 0);
    s->set_contrast(s, 2);
    // s->set_saturation(s, -2);
    s->set_exposure_ctrl(s, 0);
    s->set_ae_level(s, 2);
    s->set_aec_value(s, 1200);
    s->set_gain_ctrl(s, 0);
    s->set_agc_gain(s, 0);
    s->set_whitebal(s, 0);
    s->set_awb_gain(s, 0);

    s->set_vflip(s, 1); // flip it back
    s->set_brightness(s, 1); // up the brightness just a bit
    s->set_saturation(s, -2); // lower the saturation
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
            // if (pixel >= 192)
            //     pixel = 255;
            input_buf[k] = 255 - (float)pixel;
            k++;
        }
    }
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

void preview_input_buf()
{
    printf("XXXXXXXXXXXXXXXX ");
    for(i=0; i<784; i++)
    {
        printf("%d ", (uint8_t)input_buf[i]);
        // ESP_LOGI(TAG, "%.2f ", input_buf[i]);
    }
    printf("OOOOOOOOOOOOOOOO ");
    printf("\n");
    // ESP_LOGI(TAG,"\n");
}

void app_main(void)
{
    ESP_LOGI(TAG, "App starting...");
    memfree_at_init = esp_get_free_heap_size();
    camera_config();
    camera_init();

    ctx = uonnx_init(mnist_onnx, sizeof(mnist_onnx), mnist_planner, sizeof(mnist_planner));
    input = tensor_search(ctx->arena, "Input3");
    output = tensor_search(ctx->arena, "Plus214_Output_0");
    
    while(1)
    {
        get_frame();
        preview_input_buf();
        tensor_apply((void *)input_buf, sizeof(input_buf), input);
        uonnx_run(ctx);
        
        memfree_at_run = esp_get_free_heap_size();
        memuse_at_run = memfree_at_init - memfree_at_run;  
        memcpy(output_buf, output->datas, sizeof(output_buf));      
        output_argmax();
        print_label();
    }

    uonnx_free(ctx);
    // free_model(model);

    memfree_at_free = esp_get_free_heap_size();
    memuse_at_free = memfree_at_init - memfree_at_free;
}

// void app_main(void)
// {
//     
//     // FFD per node


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
