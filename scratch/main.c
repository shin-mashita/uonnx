#include "mtest.h"
#include "onnx_loader.h"
#include "onnx_logger.h"
#include "proto/onnx.proto3.pb-c.h"


int main()
{
    // testprint();
    // testsum(2,3);

    // Teststruct * s;
    // s = malloc(sizeof(Teststruct));
    // s = init_teststruct(s);
    // print_teststruct(s);

    // onnx_model_dump(NULL);
    Onnx__ModelProto * model = onnx_load_model_from_file("./scratch/model.onnx");
    onnx_model_dump(model);
    

    return 0;
}