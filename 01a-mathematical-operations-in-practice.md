# Mathematical Operations in Practice

## Overview

This document is a companion to [Mathematical Foundations](./01-mathematical-foundations.md). While that document introduces the **what** (definitions and properties), this document focuses on the **how** — detailed algorithms, step-by-step breakdowns, and complete C implementations for every core mathematical operation used in neural networks.

Every operation includes:
- Theoretical explanation with worked examples
- Algorithm pseudocode
- Complete, compilable C code
- Complexity analysis
- Neural network context (where/why the operation matters)

---

## Table of Contents

1. [Memory Layout and Data Structures](#1-memory-layout-and-data-structures)
2. [Vector Operations](#2-vector-operations)
3. [Matrix Operations](#3-matrix-operations)
4. [Matrix-Vector Operations](#4-matrix-vector-operations)
5. [Tensor Operations](#5-tensor-operations)
6. [Norms and Distances](#6-norms-and-distances)
7. [Matrix Decompositions](#7-matrix-decompositions)
8. [Numerical Stability Techniques](#8-numerical-stability-techniques)
9. [Broadcasting](#9-broadcasting)
10. [Putting It All Together: A Forward Pass](#10-putting-it-all-together-a-forward-pass)

---

## 1. Memory Layout and Data Structures

### 1.1 Row-Major vs Column-Major Order

Before implementing any operation, you must decide how multi-dimensional data is stored in flat (1D) memory.

**Row-Major Order** (C default):
Elements of each row are stored contiguously. For a matrix `A[m][n]`, element `A[i][j]` lives at offset `i * n + j`.

```
Matrix (conceptual):        Memory (row-major):
| 1  2  3 |                 [1, 2, 3, 4, 5, 6]
| 4  5  6 |                  ^row0^   ^row1^
```

**Column-Major Order** (Fortran, BLAS):
Elements of each column are stored contiguously. Element `A[i][j]` lives at offset `j * m + i`.

```
Matrix (conceptual):        Memory (column-major):
| 1  2  3 |                 [1, 4, 2, 5, 3, 6]
| 4  5  6 |                  ^col0^ ^col1^ ^col2^
```

**Why it matters**: Cache performance. Iterating over contiguous memory is dramatically faster due to CPU cache lines (typically 64 bytes). Always iterate over the innermost stored dimension.

### 1.2 Core Data Structures in C

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --------------------------------------------------------
 * Vector: 1D array of floats
 * -------------------------------------------------------- */
typedef struct {
    float *data;    /* heap-allocated storage            */
    int    size;    /* number of elements                */
} Vec;

Vec vec_alloc(int size) {
    Vec v;
    v.size = size;
    v.data = (float *)calloc(size, sizeof(float));
    if (!v.data) { fprintf(stderr, "vec_alloc failed\n"); exit(1); }
    return v;
}

void vec_free(Vec *v) {
    free(v->data);
    v->data = NULL;
    v->size = 0;
}

void vec_print(const Vec *v, const char *name) {
    printf("%s = [", name);
    for (int i = 0; i < v->size; i++) {
        printf("%.4f%s", v->data[i], i < v->size - 1 ? ", " : "");
    }
    printf("]\n");
}

/* --------------------------------------------------------
 * Matrix: 2D array stored in row-major order
 * -------------------------------------------------------- */
typedef struct {
    float *data;    /* heap-allocated, row-major          */
    int    rows;    /* number of rows    (m)              */
    int    cols;    /* number of columns (n)              */
} Mat;

Mat mat_alloc(int rows, int cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)calloc(rows * cols, sizeof(float));
    if (!m.data) { fprintf(stderr, "mat_alloc failed\n"); exit(1); }
    return m;
}

void mat_free(Mat *m) {
    free(m->data);
    m->data = NULL;
    m->rows = m->cols = 0;
}

/* Access element A[i][j] in row-major layout */
static inline float mat_get(const Mat *m, int i, int j) {
    return m->data[i * m->cols + j];
}

/* Set element A[i][j] */
static inline void mat_set(Mat *m, int i, int j, float val) {
    m->data[i * m->cols + j] = val;
}

void mat_print(const Mat *m, const char *name) {
    printf("%s (%dx%d) =\n", name, m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  [");
        for (int j = 0; j < m->cols; j++) {
            printf("%8.4f%s", mat_get(m, i, j), j < m->cols - 1 ? ", " : "");
        }
        printf("]\n");
    }
}

/* --------------------------------------------------------
 * Tensor: N-dimensional array (row-major, C-contiguous)
 * -------------------------------------------------------- */
#define MAX_TENSOR_DIMS 8

typedef struct {
    float *data;
    int    ndim;                    /* number of dimensions        */
    int    shape[MAX_TENSOR_DIMS];  /* size along each dimension   */
    int    stride[MAX_TENSOR_DIMS]; /* elements to skip per dim    */
    int    total;                   /* total number of elements    */
} Tensor;

Tensor tensor_alloc(int ndim, const int *shape) {
    Tensor t;
    t.ndim  = ndim;
    t.total = 1;
    for (int i = 0; i < ndim; i++) {
        t.shape[i] = shape[i];
    }
    /* Compute strides (row-major: last dim has stride 1) */
    for (int i = ndim - 1; i >= 0; i--) {
        t.stride[i] = (i == ndim - 1) ? 1 : t.stride[i + 1] * t.shape[i + 1];
    }
    for (int i = 0; i < ndim; i++) t.total *= shape[i];
    t.data = (float *)calloc(t.total, sizeof(float));
    if (!t.data) { fprintf(stderr, "tensor_alloc failed\n"); exit(1); }
    return t;
}

void tensor_free(Tensor *t) {
    free(t->data);
    t->data  = NULL;
    t->total = 0;
}

/* Flat index from N-dimensional indices */
static inline int tensor_offset(const Tensor *t, const int *indices) {
    int off = 0;
    for (int i = 0; i < t->ndim; i++) off += indices[i] * t->stride[i];
    return off;
}
```

**Stride explained**: For a 3D tensor of shape `[2, 3, 4]` (e.g., 2 images, 3 rows, 4 cols):
- `stride[2] = 1` (move one element to advance in the last dimension)
- `stride[1] = 4` (skip 4 elements to advance one row)
- `stride[0] = 12` (skip 3×4 = 12 elements to advance one "image")

---

## 2. Vector Operations

### 2.1 Vector Addition

**Theory**: Given two vectors **a**, **b** ∈ ℝⁿ, their sum **c** = **a** + **b** is defined element-wise:

```
cᵢ = aᵢ + bᵢ,   for i = 0, 1, ..., n-1
```

**Worked Example**:
```
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
c = [1.0+4.0, 2.0+5.0, 3.0+6.0] = [5.0, 7.0, 9.0]
```

**C Implementation**:

```c
/*
 * vec_add: c = a + b
 *
 * Precondition:  a.size == b.size == c.size
 * Complexity:    O(n)
 * Memory:        O(1) extra (writes into caller-provided c)
 */
void vec_add(const Vec *a, const Vec *b, Vec *c) {
    for (int i = 0; i < a->size; i++) {
        c->data[i] = a->data[i] + b->data[i];
    }
}
```

**Neural Network Context**: Bias addition in every layer: `z = Wx + b`, where adding **b** to the result of `Wx` is a vector addition.

### 2.2 Vector Subtraction

**Theory**: **c** = **a** − **b**, defined element-wise: `cᵢ = aᵢ − bᵢ`

```c
void vec_sub(const Vec *a, const Vec *b, Vec *c) {
    for (int i = 0; i < a->size; i++) {
        c->data[i] = a->data[i] - b->data[i];
    }
}
```

**Neural Network Context**: Error computation — subtracting predicted output from target: `error = target - predicted`.

### 2.3 Scalar-Vector Multiplication

**Theory**: Given scalar α and vector **a** ∈ ℝⁿ:

```
bᵢ = α · aᵢ,   for i = 0, 1, ..., n-1
```

**Worked Example**:
```
α = 2.0,  a = [3.0, -1.0, 4.0]
b = [6.0, -2.0, 8.0]
```

```c
/*
 * vec_scale: b = alpha * a
 */
void vec_scale(float alpha, const Vec *a, Vec *b) {
    for (int i = 0; i < a->size; i++) {
        b->data[i] = alpha * a->data[i];
    }
}

/*
 * vec_scale_inplace: a *= alpha  (modifies a directly)
 */
void vec_scale_inplace(float alpha, Vec *a) {
    for (int i = 0; i < a->size; i++) {
        a->data[i] *= alpha;
    }
}
```

**Neural Network Context**: Learning rate scaling — gradients are multiplied by a small scalar (learning rate) before updating weights: `w = w - lr * grad`.

### 2.4 Dot Product (Inner Product)

**Theory**: The dot product of **a**, **b** ∈ ℝⁿ produces a **scalar**:

```
a · b = Σᵢ aᵢ bᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

**Geometric interpretation**: `a · b = ‖a‖ ‖b‖ cos θ` where θ is the angle between the vectors.

- If `a · b > 0`: vectors point in roughly the same direction
- If `a · b = 0`: vectors are orthogonal (perpendicular)
- If `a · b < 0`: vectors point in roughly opposite directions

**Worked Example**:
```
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
a · b = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32.0
```

```c
/*
 * vec_dot: returns a · b
 *
 * Complexity: O(n) multiplications + O(n) additions
 *
 * NOTE: Accumulating in double avoids precision loss for
 *       large vectors. Cast back to float at the end.
 */
float vec_dot(const Vec *a, const Vec *b) {
    double sum = 0.0;
    for (int i = 0; i < a->size; i++) {
        sum += (double)a->data[i] * (double)b->data[i];
    }
    return (float)sum;
}
```

**Neural Network Context**: Each neuron computes `z = w · x + b`, which is a dot product between weight vector and input vector, plus a bias.

### 2.5 Element-wise (Hadamard) Vector Product

**Theory**: **c** = **a** ⊙ **b**, where `cᵢ = aᵢ × bᵢ`

Unlike the dot product, this returns a **vector**, not a scalar.

```c
void vec_hadamard(const Vec *a, const Vec *b, Vec *c) {
    for (int i = 0; i < a->size; i++) {
        c->data[i] = a->data[i] * b->data[i];
    }
}
```

**Neural Network Context**: Gradient of element-wise activation functions: `dz = da ⊙ σ'(z)`, where `σ'(z)` is the derivative of the activation applied element-wise.

### 2.6 Cross Product (3D only)

**Theory**: The cross product **c** = **a** × **b** is defined only for 3D vectors and produces a vector **perpendicular** to both inputs:

```
c = [a₂b₃ − a₃b₂,  a₃b₁ − a₁b₃,  a₁b₂ − a₂b₁]
```

```c
void vec_cross(const Vec *a, const Vec *b, Vec *c) {
    /* Only defined for 3D vectors */
    c->data[0] = a->data[1] * b->data[2] - a->data[2] * b->data[1];
    c->data[1] = a->data[2] * b->data[0] - a->data[0] * b->data[2];
    c->data[2] = a->data[0] * b->data[1] - a->data[1] * b->data[0];
}
```

**Neural Network Context**: Rarely used in standard neural networks, but important in 3D vision and geometric deep learning for computing surface normals.

### 2.7 Outer Product

**Theory**: Given **a** ∈ ℝᵐ and **b** ∈ ℝⁿ, the outer product produces a **matrix** C ∈ ℝᵐˣⁿ:

```
C = a bᵀ
Cᵢⱼ = aᵢ × bⱼ
```

**Worked Example**:
```
a = [1, 2, 3]ᵀ     b = [4, 5]ᵀ

C = a bᵀ = [1×4  1×5]   = [4   5 ]
           [2×4  2×5]     [8   10]
           [3×4  3×5]     [12  15]
```

```c
/*
 * vec_outer: C = a * b^T
 * Result is a (a.size × b.size) matrix.
 *
 * Complexity: O(m * n)
 */
void vec_outer(const Vec *a, const Vec *b, Mat *C) {
    for (int i = 0; i < a->size; i++) {
        for (int j = 0; j < b->size; j++) {
            mat_set(C, i, j, a->data[i] * b->data[j]);
        }
    }
}
```

**Neural Network Context**: Weight gradient computation during backpropagation: `∂L/∂W = δ · aᵀ` (output error times input activations transposed).

### 2.8 Linear Combination (AXPY)

**Theory**: `y = α·x + y` — the classic BLAS Level 1 operation. It scales a vector and adds it to another.

```c
/*
 * vec_axpy: y = alpha * x + y
 *
 * This is the backbone of gradient descent:
 *   weight = weight + (-lr) * gradient
 *   i.e., vec_axpy(-lr, &gradient, &weight)
 */
void vec_axpy(float alpha, const Vec *x, Vec *y) {
    for (int i = 0; i < x->size; i++) {
        y->data[i] += alpha * x->data[i];
    }
}
```

---

## 3. Matrix Operations

### 3.1 Matrix Addition

**Theory**: C = A + B, where all three are m × n matrices:

```
Cᵢⱼ = Aᵢⱼ + Bᵢⱼ
```

```c
void mat_add(const Mat *A, const Mat *B, Mat *C) {
    int total = A->rows * A->cols;
    for (int i = 0; i < total; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
}
```

### 3.2 Matrix Subtraction

```c
void mat_sub(const Mat *A, const Mat *B, Mat *C) {
    int total = A->rows * A->cols;
    for (int i = 0; i < total; i++) {
        C->data[i] = A->data[i] - B->data[i];
    }
}
```

### 3.3 Scalar-Matrix Multiplication

```c
void mat_scale(float alpha, const Mat *A, Mat *B) {
    int total = A->rows * A->cols;
    for (int i = 0; i < total; i++) {
        B->data[i] = alpha * A->data[i];
    }
}
```

### 3.4 Matrix Multiplication (The Core Operation)

**Theory**: Given A ∈ ℝᵐˣⁿ and B ∈ ℝⁿˣᵖ, the product C = AB ∈ ℝᵐˣᵖ is:

```
Cᵢⱼ = Σₖ Aᵢₖ Bₖⱼ,   for k = 0, 1, ..., n-1
```

Each element of C is the **dot product** of row i of A with column j of B.

**Dimensional Requirement**: The inner dimensions must match:
```
(m × n) · (n × p) → (m × p)
         ^^^
     these must match
```

**Worked Example**:
```
A (2×3):                B (3×2):
| 1  2  3 |            | 7   8 |
| 4  5  6 |            | 9  10 |
                        | 11 12 |

C = AB (2×2):

C[0][0] = 1×7 + 2×9  + 3×11 = 7  + 18 + 33 = 58
C[0][1] = 1×8 + 2×10 + 3×12 = 8  + 20 + 36 = 64
C[1][0] = 4×7 + 5×9  + 6×11 = 28 + 45 + 66 = 139
C[1][1] = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154

C = | 58   64 |
    | 139 154 |
```

#### 3.4.1 Naive Implementation (ijk)

```c
/*
 * mat_mul_naive: C = A * B
 *
 * A is (m × n), B is (n × p), C is (m × p)
 * Complexity: O(m * n * p)
 *
 * This is the textbook triple-loop. Simple but NOT cache-friendly
 * because the inner loop strides over B's columns (non-contiguous
 * in row-major layout).
 */
void mat_mul_naive(const Mat *A, const Mat *B, Mat *C) {
    int m = A->rows, n = A->cols, p = B->cols;

    /* Zero out C first */
    memset(C->data, 0, m * p * sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += mat_get(A, i, k) * mat_get(B, k, j);
            }
            mat_set(C, i, j, sum);
        }
    }
}
```

#### 3.4.2 Cache-Friendly Implementation (ikj loop order)

**Key Insight**: By reordering the loops to `i-k-j`, the inner loop iterates over columns of C and columns of B **within the same row** — both are contiguous in row-major memory.

```c
/*
 * mat_mul_ikj: C = A * B  (cache-friendly loop order)
 *
 * Inner loop walks j (column index) for both B and C,
 * accessing contiguous memory in row-major layout.
 * Typically 2-4x faster than naive ijk for large matrices.
 */
void mat_mul_ikj(const Mat *A, const Mat *B, Mat *C) {
    int m = A->rows, n = A->cols, p = B->cols;
    memset(C->data, 0, m * p * sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            float a_ik = mat_get(A, i, k);
            for (int j = 0; j < p; j++) {
                C->data[i * p + j] += a_ik * B->data[k * p + j];
            }
        }
    }
}
```

#### 3.4.3 Blocked (Tiled) Matrix Multiplication

**Theory**: Divide matrices into small blocks that fit in CPU cache (L1 cache is typically 32-64 KB). This reduces cache misses dramatically for large matrices.

```c
/*
 * mat_mul_blocked: C = A * B  (tiled for cache)
 *
 * BLOCK_SIZE should be chosen so that 3 blocks fit in L1 cache:
 *   3 * BLOCK_SIZE^2 * sizeof(float) <= L1_cache_size
 *   For 32KB L1: BLOCK_SIZE ~ 50   (we use 32 for alignment)
 */
#define BLOCK_SIZE 32

void mat_mul_blocked(const Mat *A, const Mat *B, Mat *C) {
    int m = A->rows, n = A->cols, p = B->cols;
    memset(C->data, 0, m * p * sizeof(float));

    for (int ii = 0; ii < m; ii += BLOCK_SIZE) {
        for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
            for (int jj = 0; jj < p; jj += BLOCK_SIZE) {

                /* Multiply blocks */
                int i_end = (ii + BLOCK_SIZE < m) ? ii + BLOCK_SIZE : m;
                int k_end = (kk + BLOCK_SIZE < n) ? kk + BLOCK_SIZE : n;
                int j_end = (jj + BLOCK_SIZE < p) ? jj + BLOCK_SIZE : p;

                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a_ik = A->data[i * n + k];
                        for (int j = jj; j < j_end; j++) {
                            C->data[i * p + j] += a_ik * B->data[k * p + j];
                        }
                    }
                }
            }
        }
    }
}
```

### 3.5 Matrix Transpose

**Theory**: Given A ∈ ℝᵐˣⁿ, the transpose Aᵀ ∈ ℝⁿˣᵐ swaps rows and columns:

```
(Aᵀ)ᵢⱼ = Aⱼᵢ
```

```c
/*
 * mat_transpose: B = Aᵀ
 * A is (m × n), B must be pre-allocated as (n × m).
 */
void mat_transpose(const Mat *A, Mat *B) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            mat_set(B, j, i, mat_get(A, i, j));
        }
    }
}
```

**Neural Network Context**: Backpropagation through a linear layer `y = Wx`:
- Forward: `y = W · x` (W is m×n, x is n×1)
- Backward: `∂L/∂x = Wᵀ · ∂L/∂y` — requires the transpose of W.

### 3.6 Element-wise (Hadamard) Matrix Product

```c
/*
 * mat_hadamard: C = A ⊙ B  (element-wise multiplication)
 * All matrices must have the same shape.
 */
void mat_hadamard(const Mat *A, const Mat *B, Mat *C) {
    int total = A->rows * A->cols;
    for (int i = 0; i < total; i++) {
        C->data[i] = A->data[i] * B->data[i];
    }
}
```

### 3.7 Matrix Determinant (2×2 and 3×3)

**Theory**: The determinant captures the "volume scaling factor" of a linear transformation. A matrix is invertible if and only if `det(A) ≠ 0`.

**2×2 Case**:
```
det(A) = a₁₁a₂₂ − a₁₂a₂₁

| a  b |
| c  d |  →  det = ad − bc
```

**3×3 Case** (Sarrus' rule / cofactor expansion):
```
| a  b  c |
| d  e  f |  →  det = a(ei−fh) − b(di−fg) + c(dh−eg)
| g  h  i |
```

```c
float mat_det2(const Mat *A) {
    return mat_get(A, 0, 0) * mat_get(A, 1, 1)
         - mat_get(A, 0, 1) * mat_get(A, 1, 0);
}

float mat_det3(const Mat *A) {
    float a = mat_get(A,0,0), b = mat_get(A,0,1), c = mat_get(A,0,2);
    float d = mat_get(A,1,0), e = mat_get(A,1,1), f = mat_get(A,1,2);
    float g = mat_get(A,2,0), h = mat_get(A,2,1), k = mat_get(A,2,2);
    return a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g);
}
```

### 3.8 Matrix Inverse (2×2 and General via Gauss-Jordan)

**2×2 Inverse**:
```
A⁻¹ = (1/det(A)) * | d  -b |
                     |-c   a |
```

```c
int mat_inv2(const Mat *A, Mat *Ainv) {
    float det = mat_det2(A);
    if (fabsf(det) < 1e-10f) return -1; /* singular */
    float inv_det = 1.0f / det;
    mat_set(Ainv, 0, 0,  mat_get(A,1,1) * inv_det);
    mat_set(Ainv, 0, 1, -mat_get(A,0,1) * inv_det);
    mat_set(Ainv, 1, 0, -mat_get(A,1,0) * inv_det);
    mat_set(Ainv, 1, 1,  mat_get(A,0,0) * inv_det);
    return 0;
}
```

**General n×n Inverse via Gauss-Jordan Elimination**:

The algorithm augments A with the identity matrix [A | I], then performs row operations until the left side becomes I, at which point the right side is A⁻¹: [I | A⁻¹].

```c
/*
 * mat_inv_gauss: Ainv = A⁻¹  using Gauss-Jordan elimination
 *
 * A must be square (n × n). Ainv is pre-allocated (n × n).
 * Returns 0 on success, -1 if A is singular.
 *
 * Algorithm:
 *   1. Create augmented matrix [A | I]  (n × 2n)
 *   2. For each column k = 0..n-1:
 *      a. Find pivot (row with max absolute value in column k)
 *      b. Swap pivot row with row k
 *      c. Scale row k so pivot becomes 1
 *      d. Eliminate column k in all other rows
 *   3. Right half of augmented matrix is now A⁻¹
 */
int mat_inv_gauss(const Mat *A, Mat *Ainv) {
    int n = A->rows;

    /* Augmented matrix [A | I] of size n × 2n */
    Mat aug = mat_alloc(n, 2 * n);

    /* Fill left half with A, right half with I */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat_set(&aug, i, j,     mat_get(A, i, j));
            mat_set(&aug, i, j + n, (i == j) ? 1.0f : 0.0f);
        }
    }

    /* Forward elimination with partial pivoting */
    for (int k = 0; k < n; k++) {
        /* Find pivot */
        int max_row = k;
        float max_val = fabsf(mat_get(&aug, k, k));
        for (int i = k + 1; i < n; i++) {
            float val = fabsf(mat_get(&aug, i, k));
            if (val > max_val) { max_val = val; max_row = i; }
        }
        if (max_val < 1e-10f) { mat_free(&aug); return -1; }

        /* Swap rows */
        if (max_row != k) {
            for (int j = 0; j < 2 * n; j++) {
                float tmp = mat_get(&aug, k, j);
                mat_set(&aug, k, j, mat_get(&aug, max_row, j));
                mat_set(&aug, max_row, j, tmp);
            }
        }

        /* Scale pivot row */
        float pivot = mat_get(&aug, k, k);
        for (int j = 0; j < 2 * n; j++) {
            mat_set(&aug, k, j, mat_get(&aug, k, j) / pivot);
        }

        /* Eliminate column k in all other rows */
        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            float factor = mat_get(&aug, i, k);
            for (int j = 0; j < 2 * n; j++) {
                mat_set(&aug, i, j,
                        mat_get(&aug, i, j) - factor * mat_get(&aug, k, j));
            }
        }
    }

    /* Extract inverse from right half */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat_set(Ainv, i, j, mat_get(&aug, i, j + n));
        }
    }

    mat_free(&aug);
    return 0;
}
```

---

## 4. Matrix-Vector Operations

### 4.1 Matrix-Vector Multiplication

**Theory**: Given A ∈ ℝᵐˣⁿ and **x** ∈ ℝⁿ, the product **y** = A**x** ∈ ℝᵐ is:

```
yᵢ = Σⱼ Aᵢⱼ xⱼ,   for j = 0, 1, ..., n-1
```

Each element of **y** is the dot product of a row of A with **x**.

**Worked Example**:
```
A (2×3):            x (3×1):
| 1  2  3 |        | 1 |
| 4  5  6 |        | 2 |
                    | 3 |

y[0] = 1×1 + 2×2 + 3×3 = 1 + 4 + 9  = 14
y[1] = 4×1 + 5×2 + 6×3 = 4 + 10 + 18 = 32

y = [14, 32]ᵀ
```

```c
/*
 * mat_vec_mul: y = A * x
 *
 * A is (m × n), x has n elements, y has m elements.
 * This is THE fundamental neural network operation.
 */
void mat_vec_mul(const Mat *A, const Vec *x, Vec *y) {
    for (int i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < A->cols; j++) {
            sum += (double)mat_get(A, i, j) * (double)x->data[j];
        }
        y->data[i] = (float)sum;
    }
}
```

**Neural Network Context**: This is the single most important operation. A fully-connected layer computes `y = Wx + b` — a matrix-vector multiply followed by a vector addition.

### 4.2 Affine Transformation (y = Wx + b)

```c
/*
 * affine_forward: y = W * x + b
 *
 * W is (out_dim × in_dim), x is (in_dim), b is (out_dim), y is (out_dim)
 * This computes the full linear layer forward pass.
 */
void affine_forward(const Mat *W, const Vec *x, const Vec *b, Vec *y) {
    mat_vec_mul(W, x, y);
    vec_add(y, b, y);  /* y += b (safe because same pointer for src/dst) */
}
```

---

## 5. Tensor Operations

### 5.1 Understanding Tensors

A **tensor** is the generalization of scalars, vectors, and matrices to arbitrary dimensions:

| Rank  | Name     | Example Shape  | Example Use                |
|-------|----------|----------------|----------------------------|
| 0     | Scalar   | ()             | Loss value                 |
| 1     | Vector   | (n,)           | Bias vector                |
| 2     | Matrix   | (m, n)         | Weight matrix              |
| 3     | 3D Tensor| (b, m, n)      | Batch of matrices          |
| 4     | 4D Tensor| (b, c, h, w)   | Batch of images (NCHW)     |

### 5.2 Tensor Element Access

```c
/*
 * Access element of a 3D tensor T[i][j][k]
 * Shape: (d0, d1, d2), Strides: (d1*d2, d2, 1)
 */
static inline float tensor3_get(const Tensor *t, int i, int j, int k) {
    return t->data[i * t->stride[0] + j * t->stride[1] + k];
}

static inline void tensor3_set(Tensor *t, int i, int j, int k, float v) {
    t->data[i * t->stride[0] + j * t->stride[1] + k] = v;
}
```

### 5.3 Tensor Addition (Element-wise)

**Theory**: Same as matrix/vector addition, but generalized to N dimensions. Both tensors must have the same shape.

```c
/*
 * tensor_add: C = A + B  (element-wise, any rank)
 * Precondition: A, B, C have identical shape and total count.
 */
void tensor_add(const Tensor *A, const Tensor *B, Tensor *C) {
    for (int i = 0; i < A->total; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
}
```

### 5.4 Tensor Scalar Multiply

```c
void tensor_scale(float alpha, const Tensor *A, Tensor *B) {
    for (int i = 0; i < A->total; i++) {
        B->data[i] = alpha * A->data[i];
    }
}
```

### 5.5 Tensor Element-wise Multiply (Hadamard)

```c
void tensor_hadamard(const Tensor *A, const Tensor *B, Tensor *C) {
    for (int i = 0; i < A->total; i++) {
        C->data[i] = A->data[i] * B->data[i];
    }
}
```

### 5.6 Batched Matrix Multiplication

**Theory**: Given a batch of matrices A with shape `(batch, m, n)` and B with shape `(batch, n, p)`, compute C with shape `(batch, m, p)` where each slice is an independent matrix multiply:

```
C[b] = A[b] × B[b],   for b = 0, 1, ..., batch-1
```

```c
/*
 * batched_mat_mul: C[b] = A[b] * B[b] for each batch index b
 *
 * A shape: (batch, m, n)
 * B shape: (batch, n, p)
 * C shape: (batch, m, p)
 */
void batched_mat_mul(const Tensor *A, const Tensor *B, Tensor *C) {
    int batch = A->shape[0];
    int m = A->shape[1], n = A->shape[2], p = B->shape[2];

    for (int b = 0; b < batch; b++) {
        float *a_slice = A->data + b * A->stride[0];
        float *b_slice = B->data + b * B->stride[0];
        float *c_slice = C->data + b * C->stride[0];

        for (int i = 0; i < m; i++) {
            for (int k = 0; k < n; k++) {
                float a_ik = a_slice[i * n + k];
                for (int j = 0; j < p; j++) {
                    c_slice[i * p + j] += a_ik * b_slice[k * p + j];
                }
            }
        }
    }
}
```

**Neural Network Context**: When processing a batch of inputs simultaneously, each input goes through the same weight matrix, but the computation is organized as a batched matmul for GPU parallelism.

### 5.7 Tensor Reshape

**Theory**: Reshaping changes the logical shape without moving data. The total number of elements must remain the same. For example, a (2, 6) matrix can be reshaped to (3, 4), (12,), (2, 2, 3), etc.

```c
/*
 * tensor_reshape: Change shape without copying data.
 * The total element count must remain unchanged.
 * Returns 0 on success, -1 on incompatible shape.
 */
int tensor_reshape(Tensor *t, int new_ndim, const int *new_shape) {
    int new_total = 1;
    for (int i = 0; i < new_ndim; i++) new_total *= new_shape[i];
    if (new_total != t->total) return -1;

    t->ndim = new_ndim;
    for (int i = 0; i < new_ndim; i++) t->shape[i] = new_shape[i];

    /* Recompute strides for row-major layout */
    for (int i = new_ndim - 1; i >= 0; i--) {
        t->stride[i] = (i == new_ndim - 1) ? 1 : t->stride[i+1] * t->shape[i+1];
    }
    return 0;
}
```

### 5.8 Tensor Transpose (Axis Swap)

**Theory**: Generalized transpose swaps two axes. For a 3D tensor with shape `(a, b, c)`, transposing axes 1 and 2 yields shape `(a, c, b)`.

```c
/*
 * tensor_transpose_2d: out = transpose of 2D tensor (matrix)
 * For higher-rank transposes, you swap shape and stride entries.
 */
void tensor_transpose_2d(const Tensor *in, Tensor *out) {
    int rows = in->shape[0], cols = in->shape[1];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out->data[j * rows + i] = in->data[i * cols + j];
        }
    }
}

/*
 * tensor_swap_axes: Swap two axes in-place (shape/stride only).
 * Data is NOT moved; this creates a non-contiguous view.
 * Subsequent operations must respect strides.
 */
void tensor_swap_axes(Tensor *t, int axis1, int axis2) {
    int tmp;
    tmp = t->shape[axis1];  t->shape[axis1]  = t->shape[axis2];  t->shape[axis2]  = tmp;
    tmp = t->stride[axis1]; t->stride[axis1] = t->stride[axis2]; t->stride[axis2] = tmp;
}
```

---

## 6. Norms and Distances

### 6.1 L1 Norm (Manhattan Norm)

**Theory**: Sum of absolute values: `‖x‖₁ = Σ|xᵢ|`

```c
float vec_l1_norm(const Vec *v) {
    float sum = 0.0f;
    for (int i = 0; i < v->size; i++) sum += fabsf(v->data[i]);
    return sum;
}
```

### 6.2 L2 Norm (Euclidean Norm)

**Theory**: `‖x‖₂ = √(Σ xᵢ²)` — the "length" of a vector.

```c
float vec_l2_norm(const Vec *v) {
    double sum = 0.0;
    for (int i = 0; i < v->size; i++) {
        sum += (double)v->data[i] * (double)v->data[i];
    }
    return (float)sqrt(sum);
}
```

### 6.3 Max Norm (L∞ Norm)

**Theory**: `‖x‖∞ = max|xᵢ|`

```c
float vec_linf_norm(const Vec *v) {
    float mx = 0.0f;
    for (int i = 0; i < v->size; i++) {
        float a = fabsf(v->data[i]);
        if (a > mx) mx = a;
    }
    return mx;
}
```

### 6.4 Frobenius Norm (for Matrices)

**Theory**: The "L2 norm" of a matrix, treating it as a flat vector: `‖A‖_F = √(Σᵢ Σⱼ Aᵢⱼ²)`

```c
float mat_frobenius_norm(const Mat *A) {
    double sum = 0.0;
    int total = A->rows * A->cols;
    for (int i = 0; i < total; i++) {
        sum += (double)A->data[i] * (double)A->data[i];
    }
    return (float)sqrt(sum);
}
```

**Neural Network Context**: Used in weight regularization (weight decay) and for gradient clipping: `if ‖grad‖_F > threshold, scale down`.

### 6.5 Vector Normalization

**Theory**: `x̂ = x / ‖x‖₂` — produces a unit vector (length = 1).

```c
void vec_normalize(const Vec *v, Vec *out) {
    float norm = vec_l2_norm(v);
    if (norm < 1e-12f) norm = 1e-12f; /* avoid division by zero */
    for (int i = 0; i < v->size; i++) {
        out->data[i] = v->data[i] / norm;
    }
}
```

### 6.6 Cosine Similarity

**Theory**: Measures the angle between two vectors, ignoring magnitude:

```
cos_sim(a, b) = (a · b) / (‖a‖₂ · ‖b‖₂)
```

Returns a value in [-1, 1]. 1 = identical direction, 0 = orthogonal, -1 = opposite.

```c
float vec_cosine_similarity(const Vec *a, const Vec *b) {
    float dot  = vec_dot(a, b);
    float na   = vec_l2_norm(a);
    float nb   = vec_l2_norm(b);
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / (na * nb);
}
```

---

## 7. Matrix Decompositions

### 7.1 LU Decomposition

**Theory**: Factor a square matrix A into A = LU, where L is lower-triangular and U is upper-triangular. This makes solving `Ax = b` efficient: solve `Ly = b` (forward substitution), then `Ux = y` (back substitution).

```c
/*
 * lu_decompose: Factor A = LU (in-place, with partial pivoting)
 *
 * On output:
 *   - Upper triangle of A contains U
 *   - Lower triangle of A contains L (with implicit 1s on diagonal)
 *   - perm[] records row permutations
 *
 * Returns 0 on success, -1 if singular.
 */
int lu_decompose(Mat *A, int *perm) {
    int n = A->rows;
    for (int i = 0; i < n; i++) perm[i] = i;

    for (int k = 0; k < n; k++) {
        /* Partial pivoting: find max in column k */
        float max_val = 0.0f;
        int max_idx = k;
        for (int i = k; i < n; i++) {
            float v = fabsf(mat_get(A, i, k));
            if (v > max_val) { max_val = v; max_idx = i; }
        }
        if (max_val < 1e-10f) return -1;

        /* Swap rows */
        if (max_idx != k) {
            int tmp = perm[k]; perm[k] = perm[max_idx]; perm[max_idx] = tmp;
            for (int j = 0; j < n; j++) {
                float t = mat_get(A, k, j);
                mat_set(A, k, j, mat_get(A, max_idx, j));
                mat_set(A, max_idx, j, t);
            }
        }

        /* Elimination */
        for (int i = k + 1; i < n; i++) {
            float factor = mat_get(A, i, k) / mat_get(A, k, k);
            mat_set(A, i, k, factor); /* Store L */
            for (int j = k + 1; j < n; j++) {
                mat_set(A, i, j,
                        mat_get(A, i, j) - factor * mat_get(A, k, j));
            }
        }
    }
    return 0;
}
```

---

## 8. Numerical Stability Techniques

### 8.1 Log-Sum-Exp Trick

**Problem**: Computing `log(Σ exp(xᵢ))` overflows when any `xᵢ` is large.

**Solution**: Factor out the maximum: `log(Σ exp(xᵢ)) = x_max + log(Σ exp(xᵢ − x_max))`

```c
float log_sum_exp(const Vec *x) {
    /* Step 1: Find max */
    float mx = x->data[0];
    for (int i = 1; i < x->size; i++) {
        if (x->data[i] > mx) mx = x->data[i];
    }

    /* Step 2: Compute sum of shifted exponentials */
    double sum = 0.0;
    for (int i = 0; i < x->size; i++) {
        sum += exp((double)(x->data[i] - mx));
    }

    return mx + (float)log(sum);
}
```

### 8.2 Stable Softmax

**Theory**: `softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)`. Subtract max to prevent overflow.

```c
void softmax(const Vec *input, Vec *output) {
    /* Find max for numerical stability */
    float mx = input->data[0];
    for (int i = 1; i < input->size; i++) {
        if (input->data[i] > mx) mx = input->data[i];
    }

    /* Compute exp(x - max) and sum */
    double sum = 0.0;
    for (int i = 0; i < input->size; i++) {
        output->data[i] = expf(input->data[i] - mx);
        sum += (double)output->data[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / (float)sum;
    for (int i = 0; i < input->size; i++) {
        output->data[i] *= inv_sum;
    }
}
```

### 8.3 Gradient Clipping

**Theory**: Prevents exploding gradients by capping the gradient norm.

```c
/*
 * clip_gradient_norm: If ‖grad‖₂ > max_norm, scale it down.
 * grad is modified in-place.
 */
void clip_gradient_norm(Vec *grad, float max_norm) {
    float norm = vec_l2_norm(grad);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        vec_scale_inplace(scale, grad);
    }
}
```

---

## 9. Broadcasting

### 9.1 What is Broadcasting?

**Theory**: Broadcasting is a set of rules for applying element-wise operations to tensors of different shapes. Instead of requiring identical shapes, smaller tensors are "virtually expanded" to match the larger tensor.

**Rules** (applied right-to-left across dimensions):
1. If one tensor has fewer dimensions, prepend 1s to its shape.
2. Dimensions are compatible if they are equal OR one of them is 1.
3. The dimension of size 1 is "stretched" to match the other.

**Example**:
```
A shape: (3, 4)    →  each row has 4 elements
b shape: (4,)      →  prepend 1 → (1, 4) → broadcast to (3, 4)

Result: b is added to EVERY row of A.
```

```c
/*
 * mat_add_row_broadcast: C = A + b (b is broadcast across rows)
 *
 * A is (m × n), b has n elements.
 * Each row of A gets b added element-wise.
 *
 * This is exactly what happens when adding a bias vector
 * to a batch of layer outputs.
 */
void mat_add_row_broadcast(const Mat *A, const Vec *b, Mat *C) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            mat_set(C, i, j, mat_get(A, i, j) + b->data[j]);
        }
    }
}

/*
 * mat_add_col_broadcast: C = A + b (b broadcast down columns)
 * A is (m × n), b has m elements.
 */
void mat_add_col_broadcast(const Mat *A, const Vec *b, Mat *C) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            mat_set(C, i, j, mat_get(A, i, j) + b->data[i]);
        }
    }
}
```

---

## 10. Putting It All Together: A Forward Pass

### 10.1 Complete Example: Two-Layer Neural Network Forward Pass

This combines everything we've built into a real neural network computation.

```
Architecture:
  Input:   x ∈ ℝ³     (3 features)
  Layer 1: 3 → 4       (W₁ is 4×3, b₁ is 4)   + ReLU activation
  Layer 2: 4 → 2       (W₂ is 2×4, b₂ is 2)   + Softmax output

Forward pass:
  z₁ = W₁ · x + b₁          (affine)
  a₁ = ReLU(z₁)              (activation)
  z₂ = W₂ · a₁ + b₂         (affine)
  ŷ  = softmax(z₂)           (output probabilities)
```

```c
/* ReLU activation: f(x) = max(0, x) */
void relu(const Vec *input, Vec *output) {
    for (int i = 0; i < input->size; i++) {
        output->data[i] = (input->data[i] > 0.0f) ? input->data[i] : 0.0f;
    }
}

/*
 * forward_two_layer: Complete forward pass of a 2-layer network.
 */
void forward_two_layer(
    const Mat *W1, const Vec *b1,   /* Layer 1 params */
    const Mat *W2, const Vec *b2,   /* Layer 2 params */
    const Vec *x,                   /* Input           */
    Vec *z1, Vec *a1,               /* Layer 1 intermediates */
    Vec *z2, Vec *output            /* Layer 2 + output      */
) {
    /* Layer 1: z1 = W1*x + b1, a1 = relu(z1) */
    affine_forward(W1, x, b1, z1);
    relu(z1, a1);

    /* Layer 2: z2 = W2*a1 + b2, output = softmax(z2) */
    affine_forward(W2, a1, b2, z2);
    softmax(z2, output);
}

/* ---- Demo main() ---- */
int main(void) {
    /* Input: 3 features */
    Vec x = vec_alloc(3);
    x.data[0] = 1.0f; x.data[1] = 0.5f; x.data[2] = -1.0f;

    /* Layer 1: 3 → 4 */
    Mat W1 = mat_alloc(4, 3);
    float w1_init[] = {
         0.2f,  0.3f, -0.1f,
        -0.4f,  0.1f,  0.5f,
         0.3f, -0.2f,  0.1f,
         0.1f,  0.4f, -0.3f
    };
    memcpy(W1.data, w1_init, sizeof(w1_init));

    Vec b1 = vec_alloc(4);
    b1.data[0] = 0.1f; b1.data[1] = -0.1f;
    b1.data[2] = 0.0f; b1.data[3] = 0.05f;

    /* Layer 2: 4 → 2 */
    Mat W2 = mat_alloc(2, 4);
    float w2_init[] = {
         0.5f, -0.3f,  0.2f,  0.1f,
        -0.2f,  0.4f, -0.1f,  0.3f
    };
    memcpy(W2.data, w2_init, sizeof(w2_init));

    Vec b2 = vec_alloc(2);
    b2.data[0] = 0.0f; b2.data[1] = 0.0f;

    /* Intermediate storage */
    Vec z1 = vec_alloc(4);
    Vec a1 = vec_alloc(4);
    Vec z2 = vec_alloc(2);
    Vec output = vec_alloc(2);

    /* Run forward pass */
    forward_two_layer(&W1, &b1, &W2, &b2, &x, &z1, &a1, &z2, &output);

    /* Print results */
    vec_print(&x,  "Input x ");
    vec_print(&z1, "z1 (pre-act)");
    vec_print(&a1, "a1 (ReLU)   ");
    vec_print(&z2, "z2 (pre-act)");
    vec_print(&output, "Output (softmax)");

    printf("\nPredicted class: %d (prob %.4f)\n",
           output.data[0] > output.data[1] ? 0 : 1,
           output.data[0] > output.data[1] ? output.data[0] : output.data[1]);

    /* Cleanup */
    vec_free(&x);  mat_free(&W1); vec_free(&b1);
    mat_free(&W2); vec_free(&b2);
    vec_free(&z1); vec_free(&a1); vec_free(&z2); vec_free(&output);

    return 0;
}
```

---

## Quick Reference: Complexity Table

| Operation                    | Complexity     | BLAS Level |
|------------------------------|----------------|------------|
| Vector add / sub             | O(n)           | 1          |
| Scalar-vector multiply       | O(n)           | 1          |
| Dot product                  | O(n)           | 1          |
| AXPY (y = αx + y)           | O(n)           | 1          |
| Vector norm                  | O(n)           | 1          |
| Outer product                | O(mn)          | 2          |
| Matrix-vector multiply       | O(mn)          | 2          |
| Matrix add / sub             | O(mn)          | —          |
| Matrix multiply (naive)      | O(mnp)         | 3          |
| Matrix multiply (Strassen)   | O(n^2.807)     | 3          |
| Matrix transpose              | O(mn)          | —          |
| Matrix inverse (Gauss)       | O(n³)          | —          |
| LU decomposition             | O(n³)          | —          |
| Batched matmul               | O(b·m·n·p)     | 3          |
| Softmax                      | O(n)           | —          |

---

## Summary

This document covered the **practical implementation** of every core mathematical operation needed for neural networks:

1. **Data structures**: Vec, Mat, and Tensor with row-major layout and stride-based indexing
2. **Vector ops**: add, subtract, scale, dot product, cross product, outer product, AXPY
3. **Matrix ops**: add, multiply (naive + cache-friendly + blocked), transpose, Hadamard, determinant, inverse
4. **Matrix-vector ops**: matrix-vector multiply, affine transformation
5. **Tensor ops**: element-wise ops, batched matmul, reshape, transpose
6. **Norms**: L1, L2, L∞, Frobenius, normalization, cosine similarity
7. **Decompositions**: LU decomposition with partial pivoting
8. **Numerical stability**: log-sum-exp, stable softmax, gradient clipping
9. **Broadcasting**: row and column broadcast patterns
10. **Integration**: Complete two-layer neural network forward pass

Each implementation is written in plain C with no external dependencies, suitable for direct integration into the neural network library being built.

---

**Previous**: [Mathematical Foundations](./01-mathematical-foundations.md)
**Next**: [Neural Network Fundamentals](./02-neural-network-fundamentals.md)
