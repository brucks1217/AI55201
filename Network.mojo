from random import rand, randn
from math import sqrt
from memory.unsafe import DTypePointer
import numjo as mm
from numjo import Matrix
from time import now
from math import tanh
from math import exp

struct Network:
    var _input_n: Int
    var _hidden_n_1: Int
    var _hidden_n_2: Int
    var _output_n: Int
    var _learning_r: Float32
    var _weight_1: Matrix
    var _weight_2: Matrix
    var _weight_3: Matrix
    var _bias_1: Matrix
    var _bias_2: Matrix
    var _bias_3: Matrix
    
    fn __init__(inout self, input_: Int, hidden_1: Int, hidden_2: Int, output_: Int, lr: Float32):
        self._input_n = input_
        self._hidden_n_1 = hidden_1
        self._hidden_n_2 = hidden_2
        self._output_n = output_
        self._learning_r = lr
        self._weight_1 = Matrix(0, self._input_n, self._hidden_n_1)
        self._weight_2 = Matrix(0, self._hidden_n_1, self._hidden_n_2)
        self._weight_3 = Matrix(0, self._hidden_n_2, self._output_n)
        self._bias_1 = Matrix(0, 1, self._hidden_n_1)
        self._bias_2 = Matrix(0, 1, self._hidden_n_2)
        self._bias_3 = Matrix(0, 1, self._output_n)
        

    @staticmethod
    fn softmax_1d(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.rows, A.cols)
        var row_exp_sum_mat: Matrix = Matrix(A.rows, 1)
        for i in range(A.rows):
            for j in range(A.cols):
                    B[i, j] += exp(A[i, j])
        for i in range(A.rows):
            for j in range(A.cols):
                    row_exp_sum_mat[i, 0] += B[i, j]
        for i in range(A.rows):
            for j in range(A.cols):
                B[i, j] /= row_exp_sum_mat[i, 0]
        return B
    
    @staticmethod
    fn dmse(error_output: Matrix) -> Matrix:
        let deriv_coef: Float32 = 2.0 / error_output.cols
        let deriv = error_output * Matrix(Float32(deriv_coef), error_output.rows, error_output.cols)
        return deriv
    
    fn query(inout self, inputs: Matrix, targets: Matrix, peval: Bool = False) -> Matrix:
        let output: Matrix = self.train(inputs, targets, train = False, peval=peval)
        return output

    @staticmethod
    fn relu(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.rows, A.cols)
        for i in range(B.rows):
            for j in range(B.cols):
                if A[i, j] > 0.01:
                    B[i, j] = A[i, j]
                else:
                    B[i, j] = 0.0
        return B

    @staticmethod
    fn drelu(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.rows, A.cols)
        for i in range(B.rows):
            for j in range(B.cols):
                if A[i, j] > 0.01:
                    B[i, j] = 1.0
                else:
                    B[i, j] = 0.0
        return B

    @staticmethod
    fn tanh(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.rows, A.cols)
        for i in range(A.rows):
            for j in range(A.cols):
                B[i, j] = tanh(A[i, j])
        return B

    @staticmethod
    fn dtanh(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.rows, A.cols)
        for i in range(A.rows):
            for j in range(A.cols):
                B[i, j] = 1.0 - tanh(A[i, j]) ** 2
        return B    
    fn train(inout self, inputs: Matrix, targets: Matrix, train: Bool = True, peval: Bool = False) -> Matrix:
        var hidden_n_1: Matrix = Matrix(inputs.rows, self._weight_1.cols)
        var hidden_n_2: Matrix = Matrix(hidden_n_1.rows, self._weight_2.cols)
        var error_output: Matrix = Matrix(hidden_n_2.rows, self._output_n)
        var error_output_grad: Matrix = Matrix(1, self._output_n)
        var error_hidden_n_2: Matrix = Matrix(error_output_grad.rows, self._weight_3.rows)
        var error_hidden_n_1: Matrix = Matrix(error_hidden_n_2.rows, self._weight_2.rows)
        var outputs: Matrix = Matrix(1, self._output_n)
        
        let time_now = now()
        mm.vec_mm(hidden_n_1, inputs, self._weight_1)
        hidden_n_1 = hidden_n_1 + self._bias_1
        hidden_n_1 = self.relu(hidden_n_1)
        
        mm.vec_mm(hidden_n_2, hidden_n_1, self._weight_2)
        hidden_n_2 = hidden_n_2 + self._bias_2
        hidden_n_2 = self.tanh(hidden_n_2)
        
        mm.vec_mm(outputs, hidden_n_2, self._weight_3)
        outputs = outputs + self._bias_3
        outputs = self.softmax_1d(outputs)
        
        error_output = (targets - outputs)**2
        var loss: Matrix = Matrix(1, 1)
        loss[0, 0] = mm.mean(error_output)**2
        error_output = Matrix(Float32(loss[0, 0]), error_output.rows, error_output.cols)
        error_output_grad = self.dmse(error_output)
        
        mm.vec_mm(error_hidden_n_2, error_output_grad, self._weight_3.transpose())
        mm.vec_mm(error_hidden_n_1, (error_hidden_n_2 * self.dtanh(hidden_n_2)), self._weight_2.transpose())
        
        var end_time_mat: Matrix = Matrix(1, 1)

        if train:
            self._bp(inputs, hidden_n_1, hidden_n_2, error_hidden_n_1, error_hidden_n_2, error_output_grad)
            let end_time = Float32(now() - time_now)
            end_time_mat.store[1](0, 0, end_time)
            if peval:
                return end_time_mat
            else:
                return loss
        
        let end_time = Float32(now() - time_now)
        end_time_mat.store[1](0, 0, end_time)
        
        if peval:
            return end_time_mat
        
        return outputs
        
    fn _bp(inout self, inputs: Matrix, hidden_n_1: Matrix, inputs_h2: Matrix, error_hidden_n_1: Matrix, error_hidden_n_2: Matrix, error_output_grad: Matrix):
        let hidden_n_2_drelu: Matrix = error_hidden_n_2 * self.dtanh(inputs_h2)
        let hidden_n_1_drelu: Matrix = error_hidden_n_1 * self.drelu(hidden_n_1)

        var backprop_step1: Matrix = Matrix(inputs_h2.cols, error_output_grad.cols)
        var backprop_step2: Matrix = Matrix(hidden_n_1.cols, hidden_n_2_drelu.cols)
        var backprop_step3: Matrix = Matrix(inputs.cols, hidden_n_1_drelu.cols)
        
        mm.vec_mm(backprop_step1, inputs_h2.transpose(), error_output_grad)
        mm.vec_mm(backprop_step2, hidden_n_1.transpose(), hidden_n_2_drelu)
        mm.vec_mm(backprop_step3, inputs.transpose(), hidden_n_1_drelu)

        mm.bp(self._weight_3, backprop_step1, self._learning_r)
        mm.bp(self._weight_2, backprop_step2, self._learning_r)
        mm.bp(self._weight_1, backprop_step3, self._learning_r)

        mm.bp(self._bias_3, error_output_grad, self._learning_r)
        mm.bp(self._bias_1, hidden_n_2_drelu, self._learning_r)
        mm.bp(self._bias_2, hidden_n_1_drelu, self._learning_r)
        
