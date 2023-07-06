# uONNX: A Memory-efficient ONNX inference engine for TinyML
Official code repositiory for uONNX: A Memory-efficient ONNX inference engine for TinyML. 

This is a pure C99 ONNX inference engine optimized for microcontrollers. It is built on top of an existing ONNX engine [libonnx] and it is memory optimized using the techniques employed in [TFLM]. It also makes use of a preprocessor to optimize the operators to be loaded in the microcontroller. 

## Dependencies
Install required dependencies. 

```
pip install -r requirements.txt
```

## How to run
First, create a memory planner protobuf for the model. A header file containing the generated automatically with

```
make prep MODEL=<TARGET>
```

Initialize context, and input and output tensors.

```
Context * ctx;
Tensor * input, * output
ctx = uonnx_init(model_onnx, sizeof(model_onnx), model_planner, sizeof(model_planner));
input = tensor_search(ctx->arena, "InputName");
output = tensor_search(ctx->arena, "OutputName");
```

Load data in input tensor.
```
tensor_apply((void *)input_data, sizeof(input_data), input);
```

Run session.
```
uonnx_run(ctx);
```

Free context.
```
uonnx_free(ctx);
```
## MNIST example
To run example MNIST example. 
```
make run APP=examples/benchmarks/cpu/mnist
```
then
```
./build/examples/benchmarks/cpu/mnist.out
```
## Memory Benchmarks
Run benchmarks for uONNX in different models.
### MNIST Handwritten Digit Recognition (`float32`)
```
make benchmark mnist
```
### Keyword Spotting (`float32`)
```
make benchmark kws
```
### Visual Wakeword (`float32`)
```
make benchmark vww
```
### Convolutional Reference (`float32`)
```
make benchmark ref
```

## Notes
Preprocessor and automations currently support `make=4.2.1`. Compatibility with recent `make` versions are currently in progress. 

## Related Works
1. [libonnx]
2. [TensorFlow Lite for Microcontrollers]
3. [cONNXr]


[libonnx]: https://github.com/xboot/libonnx
[TFLM]: https://github.com/tensorflow/tflite-micro
[TensorFlow Lite for Microcontrollers]:https://github.com/tensorflow/tflite-micro
[cONNXr]: https://github.com/alrevuelta/cONNXr