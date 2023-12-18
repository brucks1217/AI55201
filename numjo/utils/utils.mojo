from numjo import Matrix
from algorithm import vectorize, parallelize, vectorize_unroll
from algorithm import Static2DTileUnitFunc as Tile2DFunc
import benchmark
from memory import memset_zero, stack_allocation
from random import rand
from python import Python
from tensor import Tensor
from utils.index import Index
from memory.buffer import NDBuffer

alias type = DType.float32
alias nelts = simdwidthof[type]()

fn argmax(A: Matrix) -> Int:
    var max_idx: Int = 0
    var max_val: Float32 = A[0, 0]
    for i in range(A.rows):
        for j in range(A.cols):
            if A[i, j] > max_val:
                max_val = A[i, j]
                max_idx = j
    return max_idx

fn bp(C: Matrix, A: Matrix, B: Float32):
    for m in range(C.rows):
        for k in range(C.cols):
            C[m, k] = C[m, k] - A[m, k] * B

fn mean(A: Matrix) -> Float32:
    let mean: Float32 = sum(A) / (A.rows * A.cols)
    return mean
    
fn sum(A: Matrix) -> Float32:
    var sum: Float32 = 0.0
    for i in range(A.rows):
        for j in range(A.cols):
            sum += A[i, j]
    return sum

fn sum_row(A: Matrix, row: Int) -> Float32:
    var sum: Float32 = 0.0
    for j in range(A.cols):
        sum += A[row, j]
    return sum


fn vec_mm(C: Matrix, A: Matrix, B: Matrix): #vector

    if A.cols != B.rows:
        print("Mat Mul not possible -> A.cols: " + String(A.cols) + " != B.rows: " + String(B.rows))
        
    if C.rows != A.rows or C.cols != B.cols:
        print("Mat Mul not possible -> A.rows: " + String(A.rows) + ", A.cols: " + String(A.cols) + " and B.rows: " + String(B.rows), ", B.cols: " + String(B.cols) + " don't match C.rows: " + String(C.rows) + ", C.cols: " + String(C.cols))

    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)

fn para_mm(C: Matrix, A: Matrix, B: Matrix): # parallel

    if A.cols != B.rows:
        print("Mat Mul not possible -> A.cols: " + String(A.cols) + " != B.rows: " + String(B.rows))
        
    if C.rows != A.rows or C.cols != B.cols:
        print("Mat Mul not possible -> A.rows: " + String(A.rows) + ", A.cols: " + String(A.cols) + " and B.rows: " + String(B.rows), ", B.cols: " + String(B.cols) + " don't match C.rows: " + String(C.rows) + ", C.cols: " + String(C.cols))
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)
    parallelize[calc_row](C.rows, C.rows)


fn nai_mm(C: Matrix, A: Matrix, B: Matrix): # naive

    if A.cols != B.rows:
        print("Mat Mul not possible -> A.cols: " + String(A.cols) + " != B.rows: " + String(B.rows))
        
    if C.rows != A.rows or C.cols != B.cols:
        print("Mat Mul not possible -> A.rows: " + String(A.rows) + ", A.cols: " + String(A.cols) + " and B.rows: " + String(B.rows), ", B.cols: " + String(B.cols) + " don't match C.rows: " + String(C.rows) + ", C.cols: " + String(C.cols))

    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]



fn matmul(C: Matrix, A: Matrix, B: Matrix):
    if A.cols != B.rows:
        print("Mat Mul not possible -> A.cols: " + String(A.cols) + " != B.rows: " + String(B.rows))
        
    if C.rows != A.rows or C.cols != B.cols:
        print("Mat Mul not possible -> A.rows: " + String(A.rows) + ", A.cols: " + String(A.cols) + " and B.rows: " + String(B.rows), ", B.cols: " + String(B.cols) + " don't match C.rows: " + String(C.rows) + ", C.cols: " + String(C.cols))

    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(B.cols):
                C[m, n] += A[m, k] * B[k, n]
