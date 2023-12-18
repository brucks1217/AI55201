from memory.unsafe import Pointer
from memory import memset_zero
from random import randn, rand, seed
from algorithm import vectorize


alias type = DType.float32
alias nelts = simdwidthof[type]()

struct Matrix:
    """Simple 2D Matrix that uses Float32."""
    var data: DTypePointer[type]
    var rows: Int
    var cols: Int

    # Initialize
    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, rows: Int, cols: Int, data: DTypePointer[type]):
        self.data = DTypePointer[type].alloc(rows * cols)
        self.rows = rows
        self.cols = cols

    # Initialize with only one value
    fn __init__(inout self, owned default_value: Float32, rows: Int, cols: Int) -> None:
        self.rows = rows if rows > 0 else 1
        self.cols = cols if cols > 0 else 1
        self.data = DTypePointer[type].alloc(rows * cols)
        for i in range(rows * cols):
            self.data.store(i, default_value)

    ## Initialize with random values
    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        let data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(rows, cols, data)

    ## Initialize with random n values
    @staticmethod
    fn randn(rows: Int, cols: Int) -> Self:
        let data = DTypePointer[type].alloc(rows * cols)
        randn(data, rows * cols)
        return Self(rows, cols, data)
    
    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)
    
    fn __len__(borrowed self) -> Int:
        return self.rows * self.cols

    fn __copyinit__(inout self, other: Self) -> None:
        self.rows = other.rows
        self.cols = other.cols
        self.data = Pointer[Float32].alloc(other.rows * other.cols)
        memcpy[type](self.data, other.data, other.rows * other.cols)

    fn __moveinit__(inout self, owned other: Self) -> None:
        self.rows = other.rows
        self.cols = other.cols
        self.data = Pointer[Float32].alloc(other.rows * other.cols)
        memcpy[type](self.data, other.data, other.rows * other.cols)

    fn __lt__(borrowed self, rhs: Matrix) -> Bool:
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i, j] < rhs[i, j]:
                    return True
        return False

    fn __gt__(borrowed self, rhs: Matrix) -> Bool:
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i, j] > rhs[i, j]:
                    return True
        return False

    fn __eq__(borrowed self, rhs: Matrix) -> Bool:
        for i in range(self.rows):
            for j in range(self.cols):
                let self_val: Float32 = self[i, j]
                let rhs_val: Float32 = rhs[i, j]
                if self_val < rhs_val or self_val > rhs_val:
                    return False
        return True

    fn __ne__(borrowed self, rhs: Matrix) -> Bool:
        return not self == rhs

    fn __ge__(borrowed self, rhs: Matrix) -> Bool:
        return self > rhs or self == rhs

    fn __le__(borrowed self, rhs: Matrix) -> Bool:
        return self < rhs or self == rhs

    fn __add__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] + rhs[i, j]
        return new_matrix

    fn __pow__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] ** rhs[i, j]
        return new_matrix

    fn __sub__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] - rhs[i, j]
        return new_matrix

    fn __mul__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] * rhs[i, j]
        return new_matrix

    fn __truediv__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] / rhs[i, j]
        return new_matrix

    fn __add__(borrowed self, rhs: Float32) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] + rhs
        return new_matrix

    fn __pow__(borrowed self, rhs: Float32) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] ** rhs
        return new_matrix

    fn __sub__(borrowed self, rhs: Float32) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] - rhs
        return new_matrix

    fn __mul__(borrowed self, rhs: Float32) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] * rhs
        return new_matrix

    fn __truediv__(borrowed self, rhs: Float32) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j] / rhs
        return new_matrix

    fn pow(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = rhs[i, j] ** self[i, j]
        return new_matrix
    
    fn print_all(borrowed self) -> None:
        print("[")
        for i in range(self.rows):
            print_no_newline("    [")
            for j in range(self.cols):
                print_no_newline(self[i, j])
                if j != self.cols - 1:
                    print_no_newline(", ")
            print("]," if i != self.rows - 1 else "]")
        print("]")

    fn print_row(borrowed self, row: Int) -> None:
        print("[")
        for i in range(self.cols):
            print_no_newline(self[row, i])
        print("]")
    
    fn print_col(borrowed self, col: Int) -> None:
        print("[")
        for i in range(self.rows):
            print_no_newline(self[i, col])
        print("]")

    fn shape(borrowed self) -> None:
        print("[" + String(self.rows) + "," + String(self.cols) + "]")
    
    fn transpose(borrowed self) -> Matrix:
        """
        Transposing the matrix - e.g. [3, 2] to [2, 3] but shape doesn't matter.
        """
        var new_matrix: Matrix = Matrix(self.cols, self.rows)
        for i in range(new_matrix.rows):
            for j in range(new_matrix.cols):
                new_matrix[i, j] += self[j, i]
        return new_matrix

    fn __matmul__(borrowed self, rhs: Matrix) -> Matrix:
        var C: Matrix = Matrix(self.rows, rhs.cols)
        if self.cols != rhs.rows:
            print("Mat Mul not possible -> A.cols: " + String(self.cols) + " != B.rows: " + String(rhs.rows))
        for m in range(C.rows):
            for k in range(self.cols):
                @parameter
                fn dot[nelts : Int](n : Int):
                    C.store[nelts](m, n, C.load[nelts](m, n) + self[m, k] * rhs.load[nelts](k, n))
                vectorize[nelts, dot](C.cols)
        return C
