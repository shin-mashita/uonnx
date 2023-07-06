# Evaluation of MNIST in uONNX and onnxruntime

To evaluate the CNN model for digit recognition in onnxruntime, run 

```
python eval.py
```

To evaluate the model in uONNX, generate first the MNIST dataset to be fed. 

```
python generate_mnist.py
```

then from the main uONNX directory, run

```
make run APP=examples/tests && ./build/examples/tests.out
```