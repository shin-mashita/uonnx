#include "uonnx.h"
#include "mnist_onnx.h"
#include "mnist_dataset.h"

static int argmax(float * buf, int len)
{
    if(len <= 0)
        return -1;

    int j = 0;
    int max_idx = -1;
    float pred = buf[0];

    for (j = 0; j < len; j++)
    {
        if (buf[j] >= pred)
        {
            pred = buf[j];
            max_idx = j;
        }
    }

    return max_idx;
}

int main()
{
    int i = 0;
    int pred;
    int correct = 0, wrong = 0;
    Tensor * input_T, * output_T;
    Context * ctx;
    
    ctx = uonnx_init(mnist_onnx, sizeof(mnist_onnx), mnist_planner, sizeof(mnist_planner));
    
    input_T = tensor_search(ctx->arena, "Input3");
    output_T = tensor_search(ctx->arena, "Plus214_Output_0");

    for(i = 0; i < 10000; i++)
    {
        tensor_apply((void *)(mnist_tests[i]), sizeof(mnist_tests[i]), input_T);
        uonnx_run(ctx);
        pred = argmax(output_T->datas, 10);
        printf("Input: %d | Pred: %d | Ground Truth: %d \n", i, pred, mnist_labels[i]);
        if(pred==mnist_labels[i])
        {
            correct++;
        }
        else
        {
            wrong++;
        }
    }
    
    printf("\n\nTotal inferences done: %d | Correct: %d | Wrong: %d \n", correct+wrong, correct, wrong);
    printf("Accuracy: %f\n", (double)(correct)*100.0/((double)correct+(double)wrong));
    printf("Error: %f\n", (double)(wrong)*100.0/((double)correct+(double)wrong));
    
    uonnx_free(ctx);

    return 0;
}
