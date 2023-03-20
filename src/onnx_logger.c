#include "onnx_logger.h"

void onnx_model_dump(Onnx__ModelProto * model)
{
    int i;
    if(model)
    {
        ONNX_LOG("IR Version: v%ld\r\n", model->ir_version);
        ONNX_LOG("Producer: %s %s\r\n", model->producer_name, model->producer_version);
        ONNX_LOG("Domain: %s\r\n", model->domain);
        ONNX_LOG("Imports:\r\n");
        for(i = 0; i < model->n_opset_import; i++)
            ONNX_LOG("\t%s v%ld\r\n", (strlen(model->opset_import[i]->domain) > 0) ? model->opset_import[i]->domain : "ai.onnx", model->opset_import[i]->version);
    }
    else
    {
        ONNX_LOG("Blank model.");
    }
}