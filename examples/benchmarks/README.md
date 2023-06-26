# Benchmarks of uONNX

uONNX is benchmarked based on:
1. Inference latency (MCUs only)
2. Persistent memory (Model + Graph + Planner + Arena + Context)
3. TODO: Non-persistent memory (Operators)

## CPU
Below are the benchmark results of uONNX in three ONNX models.

|Models |Model Size (KB) | Planner protobuf Size (B)|Persistent memory (KB)|
|---|---|---|---|
|[MNIST Handwritten Digit Recognition] (`float32`)|25.8|248|119.34|
|[Keyword Spotting] (`float32`, ONNX converted)|102|445|220.34|
|[Visual Wake Word] (`float32`, ONNX converted)|892|1125|1307.9|

## MCUs
Below are the benchmark results of uONNX in three different microcontroller units in different models.

### MNIST model
|MCUs| Average Inference Latency (MegaCycles) |Parsed Model Size (KB)|Parsed Planner Size (B)|Tensor Arena Size (KB)|Graph Size (KB)| Context Size (B)| Total Persistent Memory (KB)|
|---|---|---|---|---|---|---|---|
|ESP-32|55.737| 36.812| 832| 60.532 | 3.524 | 16| 101.716|
|Raspberry Pi Pico RP2040| 157.82| 37.640|920|60.536|3.648|24|102.768|
|Artemis Redboard Apollo3| 33.402| 37.888|924|60.544|3.680|24|103.060|


[MNIST Handwritten Digit Recognition]: https://github.com/onnx/models/tree/main/vision/classification/mnist
[Keyword Spotting]: https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting
[Visual Wake Word]: https://github.com/mlcommons/tiny/tree/master/benchmark/training/visual_wake_words
[MNIST model]: https://github.com/onnx/models/tree/main/vision/classification/mnist