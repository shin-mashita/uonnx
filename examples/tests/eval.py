import onnxruntime
import numpy as np
from keras.datasets import mnist

def eval_mnist_onnx(model_path = "./mnist.onnx"):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    sess = onnxruntime.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    correct = 0
    wrong = 0

    for i in range(len(test_X)):
        input_data = np.array(test_X[i], dtype=np.float32)
        input_data = np.reshape(input_data, [1,1,28,28])

        output = sess.run(None, {input_name: input_data})
        pred = np.argmax(output)
        print("Input:", i, "| Pred:", pred, "| Ground Truth:", test_y[i])

        if(pred == test_y[i]):
            correct += 1
        else:
            wrong += 1
    
    print("Total inferences:", correct+wrong, "| Correct:", correct, "| Wrong:", wrong)
    print("Accuracy:", correct/(correct+wrong))
    print("Error:", wrong/(correct+wrong))

if __name__=='__main__':
    eval_mnist_onnx()