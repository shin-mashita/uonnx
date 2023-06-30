from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

def strprocess(array):
    array = array.flatten()
    buf = "{"
    for i in range(len(array)):
        buf += str(array[i])
        if(i == len(array)-1):
            buf += '}'
        else:
            buf += ', '
    return buf

def generate_header(dataset, labels):
    buf = 'static const float mnist_tests[][784] = {\n'

    for i in range(len(dataset)):
        buf += strprocess(dataset[i])
        if(i == len(dataset)-1):
            buf += '};'
        else:
            buf += ', '

    buf += '\n\n'

    buf += 'static const int mnist_labels[] = {'

    for i in range(len(labels)):
        buf += str(labels[i])
        if(i == len(labels)-1):
            buf += '};'
        else:
            buf += ', '

    return buf

if __name__=='__main__':
    with open("mnist_dataset.h", "w") as f:
        f.write(generate_header(test_X, test_y))
