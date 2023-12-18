import benchmark
from memory import memset_zero, stack_allocation
from random import rand
from algorithm import vectorize, parallelize, vectorize_unroll
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from python import Python
from tensor import Tensor
from utils.index import Index
from memory.buffer import NDBuffer

alias M = 512
alias N = 512
alias K = 4096
alias type = DType.float32


struct Matrix:
    var data: DTypePointer[type]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    fn __init__(inout self, rows: Int, cols: Int, data: DTypePointer[DType.float32]):
        self.data = data
        self.rows = rows
        self.cols = cols

    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        let data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(rows, cols, data)

    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)


def run_matmul_python() -> Float64:
    Python.add_to_path(".")
    let pymatmul: PythonObject = Python.import_module("pymatmul")
    let py = Python.import_module("builtins")

    let gflops = pymatmul.benchmark_matmul_python(128, 128, 128).to_float64()
    py.print(py.str("{:<13}{:>8.3f} GFLOPS").format("Python:", gflops))

    return gflops


def run_matmul_numpy() -> Float64:
    let pymatmul: PythonObject = Python.import_module("pymatmul")
    let py = Python.import_module("builtins")

    let gflops = pymatmul.benchmark_matmul_numpy(M, N, K).to_float64()
    py.print(py.str("{:<13}{:>8.3f} GFLOPS").format("Numpy:", gflops))

    return gflops


fn naive(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


alias nelts = simdwidthof[type]()  # The SIMD vector width.


fn vectorized(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)


fn parallelized(inout C: Matrix, A: Matrix, B: Matrix):
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


fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


fn tiled(inout C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[nelts, dot](tile_x)

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](C.cols, B.rows)

    parallelize[calc_row](C.rows, C.rows)


fn unrolled(inout C: Matrix, A: Matrix, B: Matrix):
    alias tile_size = 4

    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        tile[calc_tile, nelts * tile_size, tile_size](C.cols, B.rows)

    parallelize[calc_row](C.rows, C.rows)


fn tile_parallel[
    tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int
](end_x: Int, end_y: Int):
    @parameter
    fn row(yo: Int):
        let y = tile_y * yo
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

    parallelize[row](end_y // tile_y, M)


fn reordered(inout C: Matrix, A: Matrix, B: Matrix):
    alias tile_k = 8
    alias tile_k_unroll = 8
    alias tile_i = 32
    alias tile_j = nelts * 4

    @parameter
    fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):
        var accumulators = Matrix(
            tile_i, tile_j, stack_allocation[tile_i * tile_j, DType.float32]()
        )

        for ko in range(0, A.cols, tile_k * tile_k_unroll):
            for _ in range(tile_i):
                for i in range(tile_k):

                    @unroll
                    for k in range(tile_k_unroll):

                        @parameter
                        fn calc_tile_cols[nelts: Int](j: Int):
                            accumulators.store[nelts](
                                i,
                                j,
                                accumulators.load[nelts](i, j)
                                + A[io + i, ko + k] * B.load[nelts](ko + k, jo + j),
                            )

                        vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

        for i in range(tile_i):
            for j in range(tile_j):
                C[io + i, jo + j] = accumulators[i, j]

    tile_parallel[calc_tile, tile_j, tile_i](C.cols, C.rows)


fn tile_parallel_swizzled[
    tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int
](end_x: Int, end_y: Int):
    alias tile_outer = 8
    alias group_size = tile_outer * 4

    @parameter
    fn row(swizzled: Int):
        let group_id = swizzled // group_size
        let group_offset_x = (group_id * tile_outer) % (N // tile_y)
        let yo = (swizzled % group_size) // tile_outer
        let xo = group_offset_x + (swizzled % tile_outer)
        let y = tile_y * yo
        let x = tile_x * xo
        tiled_fn[tile_x, tile_y](x, y)

    parallelize[row]((end_y // tile_y * end_x // tile_x), M * 2)


fn swizzled(inout C: Matrix, A: Matrix, B: Matrix):
    alias tile_k = 8
    alias tile_k_unroll = 8
    alias tile_i = 32
    alias tile_j = nelts * 4

    @parameter
    fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):
        var accumulators = Matrix(
            tile_i, tile_j, stack_allocation[tile_i * tile_j, DType.float32]()
        )

        for ko in range(0, A.cols, tile_k * tile_k_unroll):
            for _ in range(tile_i):
                for i in range(tile_k):

                    @unroll
                    for k in range(tile_k_unroll):

                        @parameter
                        fn calc_tile_cols[nelts: Int](j: Int):
                            accumulators.store[nelts](
                                i,
                                j,
                                accumulators.load[nelts](i, j)
                                + A[io + i, ko + k] * B.load[nelts](ko + k, jo + j),
                            )

                        vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

        for i in range(tile_i):
            for j in range(tile_j):
                C[io + i, jo + j] = accumulators[i, j]

    tile_parallel_swizzled[calc_tile, tile_j, tile_i](C.cols, C.rows)


@always_inline
fn bench[
    func: fn (inout Matrix, Matrix, Matrix) -> None, name: StringLiteral
](base_gflops: Float64, numpy_gflops: Float64) raises:
    var A = Matrix.rand(M, K)
    var B = Matrix.rand(K, N)
    var C = Matrix(M, N)

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    let secs = benchmark.run[test_fn]().mean()
    A.data.free()
    B.data.free()
    C.data.free()
    let gflops = ((2 * M * N * K) / secs) / 1e9
    let speedup: Float64 = gflops / base_gflops
    let numpy_speedup: Float64 = gflops / numpy_gflops

    let py = Python.import_module("builtins")
    _ = py.print(
        py.str("{:<13}{:>8.3f} GFLOPS {:>9.2f}x Python {:>5.2f}x Numpy").format(
            name, gflops, speedup, numpy_speedup
        )
    )


@always_inline
fn test[
    func: fn (inout Matrix, Matrix, Matrix) -> None
](A: Matrix, B: Matrix) raises -> SIMD[type, 1]:
    var C = Matrix(M, N)
    _ = func(C, A, B)
    var result = SIMD[type, 1]()
    for i in range(C.rows):
        for j in range(C.cols):
            result += C[i, j]
    return result


fn test_all() raises:
    constrained[M == N, "M and N must be equal for matrix multiplication"]()

    let A = Matrix.rand(M, K)
    let B = Matrix.rand(K, N)

    let result = test[naive](A, B)

    if test[vectorized](A, B) != result:
        raise Error("Vectorize output does not match")
    if test[parallelized](A, B) != result:
        raise Error("Parallelize output incorrect")
    if test[tiled](A, B) != result:
        raise Error("Tiled output incorrect")
    if test[unrolled](A, B) != result:
        raise Error("Unroll output incorrect")
    if test[reordered](A, B) != result:
        raise Error("Loop reorder output incorrect")
    if test[swizzled](A, B) != result:
        raise Error("Swizzled output incorrect")

    A.data.free()
    B.data.free()


fn main() raises:
    print("CPU Results\n")
    let python_gflops = run_matmul_python()
    let numpy_gflops = run_matmul_numpy()

    bench[naive, "Naive:"](python_gflops, numpy_gflops)
    bench[vectorized, "Vectorized:"](python_gflops, numpy_gflops)
    bench[parallelized, "Parallelized:"](python_gflops, numpy_gflops)
    bench[tiled, "Tiled:"](python_gflops, numpy_gflops)
    bench[unrolled, "Unrolled:"](python_gflops, numpy_gflops)
    bench[reordered, "Reordered:"](python_gflops, numpy_gflops)
    bench[swizzled, "Swizzled:"](python_gflops, numpy_gflops)
