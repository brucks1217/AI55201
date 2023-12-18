from Network import Network
from numjo import Matrix
from python import Python
from time import now
import numjo as mm
from random import randn, rand


fn main() raises:

    Python.add_to_path("./")
    let DataLoader = Python.import_module("DataLoader")
    let np = Python.import_module("numpy")
    var labels = np.array
    var inputs = np.array
    let labels_: Matrix = Matrix(10000, 10)
    let inputs_: Matrix = Matrix(10000, 784)
    let input_ = 784
    let hidden_1 = 150
    let hidden_2 = 80
    let output_ = 10
    let lr = 1e-4
    var new_input: Matrix = Matrix(1, input_)
    var new_label: Matrix = Matrix(1, output_)
    var peval_nn = Network(input_=input_, hidden_1=hidden_1, hidden_2=hidden_2, output_=output_, lr=lr)

    
    inputs = DataLoader.mnist_inputs("")
    labels = DataLoader.mnist_labels("", output_)

    
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            inputs_[i, j] = inputs[i][j].to_float64().cast[DType.float32]()
            if inputs_[i, j] <= 0.01:
                inputs_[i,j] = 0.0
    
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            labels_[i, j] = labels[i][j].to_float64().cast[DType.float32]()
    


    var iter_time: Matrix = Matrix(1, 1)
    var time_sum: Float32 = 0.0
    var time_now = now()
        
    for i in range(100):
            for j in range(inputs_.cols):
                new_input[0, j] = inputs_[i, j]
                if j <= 9:
                    new_label[0, j] = labels_[i, j]
        iter_time = peval_nn.train(new_input, new_label, peval=True)
    print("Runtime : " + String((now() - time_now) / 1e9) + " seconds")
    
