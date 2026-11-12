# Tutorial 01: Building Your First Tensor Class

> **Learning Goal**: By the end of this tutorial, you'll have built a complete, production-ready Tensor class in C++ that handles dynamic multi-dimensional arrays with automatic memory management. This is the foundation of your neural network library.

## Why Start With a Tensor Class?

Before we dive into neural networks, we need a solid foundation. In the world of deep learning, **everything is a tensor**:
- Scalars are 0D tensors
- Vectors are 1D tensors  
- Matrices are 2D tensors
- Higher-dimensional data (like images or video) are 3D, 4D, or higher

**Think of it this way**: If neural networks are buildings, tensors are the bricks. Without good bricks, your building will crumble.

### What We'll Build

We're going to create a `Tensor<T>` class that:
1. Stores multi-dimensional data efficiently
2. Supports both CPU operations (we'll add GPU later)
3. Handles memory automatically (RAII principles)
4. Provides mathematical operations (add, multiply, etc.)
5. Integrates with modern C++ features (move semantics, perfect forwarding)

Let's start simple and build up incrementally.

---

## Part 1: The Simplest Tensor (1D Vector)

### Step 1.1: Basic Structure

Let's start with something you already understand - a simple 1D vector. Create a new file `tensor.hpp`:

```cpp
#pragma once
#include <memory>
#include <vector>
#include <stdexcept>
#include <initializer_list>

namespace nn {

template<typename T>
class Tensor {
private:
    std::unique_ptr<T[]> data_;
    size_t size_;
    
public:
    // Constructor for 1D tensor
    explicit Tensor(size_t size) 
        : data_(std::make_unique<T[]>(size))
        , size_(size) 
    {
        // Initialize to zero
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = T{0};
        }
    }
    
    // Access elements
    T& operator[](size_t idx) {
        if (idx >= size_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[idx];
    }
    
    const T& operator[](size_t idx) const {
        if (idx >= size_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[idx];
    }
    
    size_t size() const { return size_; }
};

} // namespace nn
```

**Let's break this down:**

1. **`std::unique_ptr<T[]>`**: We use a unique pointer to manage our array. Why?
   - Automatic memory cleanup (RAII)
   - No manual `delete[]` needed
   - Move-only semantics (prevents accidental copies)

2. **`size_`**: We store the size separately. We'll need this for bounds checking.

3. **Constructor**: We initialize all elements to zero. This is important for numerical stability.

4. **`operator[]`**: Provides array-like access with bounds checking.

### Step 1.2: Let's Test It

Create `main.cpp`:

```cpp
#include "tensor.hpp"
#include <iostream>

int main() {
    // Create a 1D tensor with 5 elements
    nn::Tensor<float> vec(5);
    
    // Fill with data
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<float>(i) * 2.0f;
    }
    
    // Print
    std::cout << "Vector: [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    return 0;
}
```

**Compile and run:**
```bash
g++ -std=c++17 -Wall -Wextra main.cpp -o tensor_test
./tensor_test
```

**Expected output:**
```
Vector: [0, 2, 4, 6, 8]
```

### 🎯 Challenge 1.1
Add an initializer list constructor so you can write:
```cpp
nn::Tensor<float> vec = {1.0f, 2.0f, 3.0f};
```

**Hint:** Look up `std::initializer_list<T>` in C++ reference.

---

## Part 2: Adding Dimensions (The Shape)

Now comes the interesting part. A real tensor needs to support multiple dimensions. 

### Step 2.1: Understanding Shape

A **shape** describes the dimensions of a tensor:
- `[5]` = 1D vector with 5 elements
- `[3, 4]` = 2D matrix with 3 rows, 4 columns
- `[2, 3, 4]` = 3D tensor (imagine a stack of 2 matrices, each 3x4)

**Key insight:** Regardless of dimensions, we can store everything in a **1D array** and use clever indexing!

For a 2D matrix `[3, 4]`:
```
Row 0: indices 0,  1,  2,  3
Row 1: indices 4,  5,  6,  7
Row 2: indices 8,  9, 10, 11
```

The formula: `index = row * num_cols + col`

For 3D and higher, we generalize this with **strides**.

### Step 2.2: Implementing Shape

Update `tensor.hpp`:

```cpp
#pragma once
#include <memory>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <initializer_list>

namespace nn {

template<typename T>
class Tensor {
private:
    std::unique_ptr<T[]> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t total_size_;
    
    // Calculate strides from shape
    static std::vector<size_t> calculate_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        if (shape.empty()) return strides;
        
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
    
    // Calculate total size from shape
    static size_t calculate_total_size(const std::vector<size_t>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 
                             size_t{1}, std::multiplies<size_t>());
    }
    
public:
    // Constructor with shape
    explicit Tensor(const std::vector<size_t>& shape)
        : shape_(shape)
        , strides_(calculate_strides(shape))
        , total_size_(calculate_total_size(shape))
        , data_(std::make_unique<T[]>(total_size_))
    {
        // Initialize to zero
        std::fill_n(data_.get(), total_size_, T{0});
    }
    
    // Convenience constructor for 1D
    explicit Tensor(size_t size) 
        : Tensor(std::vector<size_t>{size}) {}
    
    // Multi-dimensional access
    T& at(const std::vector<size_t>& indices) {
        return data_[flat_index(indices)];
    }
    
    const T& at(const std::vector<size_t>& indices) const {
        return data_[flat_index(indices)];
    }
    
    // 1D access (for convenience)
    T& operator[](size_t idx) {
        if (idx >= total_size_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[idx];
    }
    
    const T& operator[](size_t idx) const {
        if (idx >= total_size_) {
            throw std::out_of_range("Index out of bounds");
        }
        return data_[idx];
    }
    
    // Getters
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t total_size() const { return total_size_; }
    size_t ndim() const { return shape_.size(); }
    
private:
    size_t flat_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
        }
        
        size_t flat = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds in dimension " + std::to_string(i));
            }
            flat += indices[i] * strides_[i];
        }
        return flat;
    }
};

} // namespace nn
```

**Understanding Strides:**

For shape `[2, 3, 4]`:
- Stride for dim 0: `3 * 4 = 12` (jump by 12 to get to next "row" of the 3D tensor)
- Stride for dim 1: `4` (jump by 4 to get to next row in a matrix)
- Stride for dim 2: `1` (jump by 1 to get to next element)

To access `[i, j, k]`: `index = i*12 + j*4 + k*1`

### Step 2.3: Testing Multi-Dimensional Access

```cpp
#include "tensor.hpp"
#include <iostream>

int main() {
    // Create a 2x3 matrix
    nn::Tensor<float> matrix({2, 3});
    
    // Fill with data
    float value = 1.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            matrix.at({i, j}) = value++;
        }
    }
    
    // Print
    std::cout << "Matrix (2x3):\n";
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            std::cout << matrix.at({i, j}) << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}
```

**Expected output:**
```
Matrix (2x3):
1 2 3
4 5 6
```

### 🎯 Challenge 2.1
Create a 3D tensor with shape `[2, 3, 4]` and fill it with values 1 through 24. Print it in a readable format showing the structure.

---

## Part 3: The Big Five (Memory Management Done Right)

We have `std::unique_ptr` managing our memory, but we're missing some important C++ operations. Let's implement the **Rule of Five**.

### Step 3.1: Why Do We Need This?

```cpp
nn::Tensor<float> a({1000});
nn::Tensor<float> b = a;  // ❌ Won't compile! unique_ptr is not copyable
```

We need to explicitly decide:
1. Should tensors be copyable? (Yes, but it's expensive)
2. Should tensors be movable? (Yes, and it's cheap!)

### Step 3.2: Implementing Copy Semantics

Add to `Tensor` class:

```cpp
// Copy constructor - deep copy
Tensor(const Tensor& other)
    : shape_(other.shape_)
    , strides_(other.strides_)
    , total_size_(other.total_size_)
    , data_(std::make_unique<T[]>(other.total_size_))
{
    std::copy_n(other.data_.get(), total_size_, data_.get());
}

// Copy assignment
Tensor& operator=(const Tensor& other) {
    if (this != &other) {
        // Create temp and swap (copy-and-swap idiom)
        Tensor temp(other);
        swap(temp);
    }
    return *this;
}

// Swap helper
void swap(Tensor& other) noexcept {
    using std::swap;
    swap(data_, other.data_);
    swap(shape_, other.shape_);
    swap(strides_, other.strides_);
    swap(total_size_, other.total_size_);
}
```

### Step 3.3: Implementing Move Semantics

```cpp
// Move constructor
Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_))
    , shape_(std::move(other.shape_))
    , strides_(std::move(other.strides_))
    , total_size_(other.total_size_)
{
    other.total_size_ = 0;
}

// Move assignment
Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        total_size_ = other.total_size_;
        other.total_size_ = 0;
    }
    return *this;
}
```

**Why is move cheap and copy expensive?**

- **Copy**: Allocates new memory and copies every element `O(n)`
- **Move**: Just transfers ownership of pointers `O(1)`

### 🎯 Challenge 3.1
Write a function that takes a `Tensor` by value. Call it twice: once with an lvalue and once with an rvalue. Use a debugger or print statements to verify when copy vs move happens.

---

## Part 4: Mathematical Operations (The Fun Part!)

Now let's make our tensor actually do math!

### Step 4.1: Element-wise Addition

```cpp
// In Tensor class
Tensor operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shapes must match for addition");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor& operator+=(const Tensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shapes must match for addition");
    }
    
    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}
```

### Step 4.2: Scalar Operations

```cpp
// Scalar multiplication
Tensor operator*(T scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

// Friend function for scalar * tensor
friend Tensor operator*(T scalar, const Tensor& tensor) {
    return tensor * scalar;
}
```

### Step 4.3: Testing Math Operations

```cpp
int main() {
    nn::Tensor<float> a({2, 2});
    nn::Tensor<float> b({2, 2});
    
    a.at({0, 0}) = 1; a.at({0, 1}) = 2;
    a.at({1, 0}) = 3; a.at({1, 1}) = 4;
    
    b.at({0, 0}) = 5; b.at({0, 1}) = 6;
    b.at({1, 0}) = 7; b.at({1, 1}) = 8;
    
    auto c = a + b;
    auto d = a * 2.0f;
    
    std::cout << "a + b:\n";
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::cout << c.at({i, j}) << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}
```

### 🎯 Challenge 4.1
Implement these operations:
1. Element-wise subtraction (`operator-`)
2. Element-wise multiplication (Hadamard product, `operator*` for two tensors)
3. Element-wise division

---

## Part 5: Matrix Multiplication (The Heart of Neural Networks)

This is the **most important** operation for neural networks. Let's implement it correctly.

### Step 5.1: Understanding Matrix Multiplication

For matrices A(`m × n`) and B(`n × p`), result C(`m × p`):

```
C[i][j] = Σ(A[i][k] * B[k][j]) for k = 0 to n-1
```

**Key constraints:**
- A must be 2D
- B must be 2D  
- A's columns must equal B's rows

### Step 5.2: Implementation (Naive)

```cpp
// Matrix multiplication (only for 2D tensors)
Tensor matmul(const Tensor& other) const {
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    
    size_t m = shape_[0];  // rows of A
    size_t n = shape_[1];  // cols of A / rows of B
    size_t p = other.shape_[1];  // cols of B
    
    if (n != other.shape_[0]) {
        throw std::invalid_argument("Inner dimensions must match");
    }
    
    Tensor result({m, p});
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            T sum = 0;
            for (size_t k = 0; k < n; ++k) {
                sum += at({i, k}) * other.at({k, j});
            }
            result.at({i, j}) = sum;
        }
    }
    
    return result;
}
```

**Complexity**: O(m × n × p) - This is the bottleneck in neural networks!

### Step 5.3: Testing Matrix Multiplication

```cpp
int main() {
    // Create a 2x3 matrix
    nn::Tensor<float> A({2, 3});
    A.at({0, 0}) = 1; A.at({0, 1}) = 2; A.at({0, 2}) = 3;
    A.at({1, 0}) = 4; A.at({1, 1}) = 5; A.at({1, 2}) = 6;
    
    // Create a 3x2 matrix
    nn::Tensor<float> B({3, 2});
    B.at({0, 0}) = 7;  B.at({0, 1}) = 8;
    B.at({1, 0}) = 9;  B.at({1, 1}) = 10;
    B.at({2, 0}) = 11; B.at({2, 1}) = 12;
    
    // Result should be 2x2
    auto C = A.matmul(B);
    
    std::cout << "A × B:\n";
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::cout << C.at({i, j}) << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}
```

**Expected output:**
```
A × B:
58 64
139 154
```

**Verification by hand:**
- C[0][0] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58 ✓
- C[0][1] = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64 ✓

### 🎯 Challenge 5.1 (Advanced)
Optimize matrix multiplication using:
1. Loop reordering for better cache locality
2. Blocked/tiled matrix multiplication
3. SIMD instructions (use intrinsics or compiler auto-vectorization)

Research: Why is the innermost loop ordering important for performance?

---

## Part 6: Broadcasting (Advanced Topic)

In frameworks like NumPy and PyTorch, you can add arrays of different shapes:

```python
a = np.array([[1, 2, 3]])      # Shape (1, 3)
b = np.array([[1], [2], [3]])  # Shape (3, 1)
c = a + b                       # Shape (3, 3)
```

This is called **broadcasting**. It's incredibly useful but requires careful implementation.

### Step 6.1: Broadcasting Rules

Two dimensions are compatible when:
1. They are equal, OR
2. One of them is 1

Examples:
- `(3, 4)` + `(3, 4)` → `(3, 4)` ✓
- `(3, 1)` + `(1, 4)` → `(3, 4)` ✓
- `(3, 4)` + `(3, 1)` → `(3, 4)` ✓
- `(3, 4)` + `(2, 4)` → Error ✗

### Step 6.2: Implementation Sketch

```cpp
// Check if shapes are broadcastable
static bool can_broadcast(const std::vector<size_t>& shape1,
                         const std::vector<size_t>& shape2,
                         std::vector<size_t>& result_shape) {
    size_t max_dims = std::max(shape1.size(), shape2.size());
    result_shape.resize(max_dims);
    
    for (size_t i = 0; i < max_dims; ++i) {
        size_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;
        
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;  // Not broadcastable
        }
        
        result_shape[max_dims - 1 - i] = std::max(dim1, dim2);
    }
    
    return true;
}
```

### 🎯 Challenge 6.1 (Very Advanced)
Implement a fully functional broadcasting addition. You'll need to:
1. Determine the result shape
2. For each output element, calculate the corresponding input indices
3. Handle dimension expansion properly

**Hint**: This is genuinely challenging. Consider it a capstone project!

---

## Part 7: Memory Layout and Performance

### Step 7.1: Row-Major vs Column-Major

We've been using **row-major** order (C-style):
```
Matrix[2][3] stored as: [row0col0, row0col1, row0col2, row1col0, row1col1, row1col2]
```

**Column-major** (Fortran-style) stores columns contiguously:
```
Matrix[2][3] stored as: [row0col0, row1col0, row0col1, row1col1, row0col2, row1col2]
```

**Why does this matter?**

Performance! When you access `A[i][:]` (entire row) in row-major, elements are contiguous in memory. Good cache locality!

### Step 7.2: Memory Alignment

Modern CPUs work best with aligned memory:

```cpp
// In Tensor constructor
Tensor(const std::vector<size_t>& shape)
    : shape_(shape)
    , strides_(calculate_strides(shape))
    , total_size_(calculate_total_size(shape))
{
    // Allocate aligned memory
    void* ptr = nullptr;
    size_t alignment = 64;  // Cache line size
    size_t size_bytes = total_size_ * sizeof(T);
    
    #ifdef _WIN32
        ptr = _aligned_malloc(size_bytes, alignment);
    #else
        if (posix_memalign(&ptr, alignment, size_bytes) != 0) {
            ptr = nullptr;
        }
    #endif
    
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    data_.reset(static_cast<T*>(ptr));
    
    // Custom deleter
    // ... (exercise for the reader!)
}
```

### 🎯 Challenge 7.1
Implement a custom deleter for aligned memory in the `unique_ptr`. Research `std::unique_ptr` with custom deleters.

---

## Part 8: Putting It All Together

Let's create a complete example that demonstrates everything:

```cpp
#include "tensor.hpp"
#include <iostream>
#include <iomanip>

void print_tensor(const nn::Tensor<float>& t, const std::string& name) {
    std::cout << name << " (shape: ";
    for (size_t dim : t.shape()) {
        std::cout << dim << " ";
    }
    std::cout << "):\n";
    
    if (t.ndim() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < t.shape()[0]; ++i) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                     << t.at({i});
            if (i < t.shape()[0] - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    } else if (t.ndim() == 2) {
        for (size_t i = 0; i < t.shape()[0]; ++i) {
            std::cout << "[";
            for (size_t j = 0; j < t.shape()[1]; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                         << t.at({i, j});
                if (j < t.shape()[1] - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== Neural Network Tensor Demo ===\n\n";
    
    // Simulate a simple neural network layer
    // Input: 3 samples, 4 features each
    nn::Tensor<float> input({3, 4});
    input.at({0, 0}) = 1.0; input.at({0, 1}) = 2.0; 
    input.at({0, 2}) = 3.0; input.at({0, 3}) = 4.0;
    input.at({1, 0}) = 5.0; input.at({1, 1}) = 6.0; 
    input.at({1, 2}) = 7.0; input.at({1, 3}) = 8.0;
    input.at({2, 0}) = 9.0; input.at({2, 1}) = 10.0; 
    input.at({2, 2}) = 11.0; input.at({2, 3}) = 12.0;
    
    print_tensor(input, "Input");
    
    // Weights: 4 input features -> 2 output features
    nn::Tensor<float> weights({4, 2});
    weights.at({0, 0}) = 0.1; weights.at({0, 1}) = 0.2;
    weights.at({1, 0}) = 0.3; weights.at({1, 1}) = 0.4;
    weights.at({2, 0}) = 0.5; weights.at({2, 1}) = 0.6;
    weights.at({3, 0}) = 0.7; weights.at({3, 1}) = 0.8;
    
    print_tensor(weights, "Weights");
    
    // Bias: 2 output features
    nn::Tensor<float> bias({1, 2});
    bias.at({0, 0}) = 1.0;
    bias.at({0, 1}) = 2.0;
    
    print_tensor(bias, "Bias");
    
    // Forward pass: output = input @ weights + bias
    auto output = input.matmul(weights);
    std::cout << "After matmul:\n";
    print_tensor(output, "Output (before bias)");
    
    // Note: This is where broadcasting would help!
    // For now, we'll add bias manually
    for (size_t i = 0; i < output.shape()[0]; ++i) {
        for (size_t j = 0; j < output.shape()[1]; ++j) {
            output.at({i, j}) += bias.at({0, j});
        }
    }
    
    print_tensor(output, "Output (after bias)");
    
    std::cout << "=== Demo Complete ===\n";
    return 0;
}
```

This demonstrates a **real neural network layer computation**: `output = input × weights + bias`

---

## Summary and Next Steps

### What You've Built

Congratulations! You now have a working tensor class with:
- ✅ Multi-dimensional arrays
- ✅ Automatic memory management
- ✅ Element-wise operations
- ✅ Matrix multiplication
- ✅ Proper C++ semantics (Rule of Five)

### What's Missing (For Now)

- Broadcasting (we discussed it)
- GPU support (coming in later tutorials)
- Automatic differentiation (the magic of backpropagation)
- Optimized operations (BLAS integration)

### Performance Reality Check

Our current implementation is **educational**, not production-ready. Here's why:

1. **No BLAS**: Professional libraries use highly optimized BLAS (Basic Linear Algebra Subprograms)
2. **No SIMD**: We're not using vector instructions
3. **Cache inefficiency**: Our matrix multiplication is naive
4. **No parallelization**: Multi-core CPUs are underutilized

**But that's okay!** Understanding the fundamentals is crucial. Optimization comes after correctness.

---

## 🎯 Final Project: Neural Network Inference

**Task**: Use your tensor class to implement a simple 2-layer neural network that can classify the XOR problem:

```
Input  | Output
-------+-------
0, 0   | 0
0, 1   | 1
1, 0   | 1
1, 1   | 0
```

**Requirements**:
1. Network structure: Input(2) → Hidden(4) → Output(1)
2. Use pre-trained weights (I'll provide them)
3. Implement sigmoid activation: `σ(x) = 1 / (1 + e^(-x))`
4. Test all four XOR cases

**Pre-trained weights** (these solve XOR):
```
W1 (2x4):
[[ 10, -10,  10, -10],
 [ 10,  10, -10, -10]]

b1 (1x4):
[[-5, 15, -15, 5]]

W2 (4x1):
[[ 10],
 [ 10],
 [ 10],
 [ 10]]

b2 (1x1):
[[-15]]
```

**Hint**: The forward pass is:
```
hidden = sigmoid(input @ W1 + b1)
output = sigmoid(hidden @ W2 + b2)
```

---

## Further Reading

1. **C++ Memory Management**:
   - "Effective Modern C++" by Scott Meyers (Items 18-25)
   - Your own blog on manual object lifetime management!

2. **Numerical Computing**:
   - "Numerical Recipes" - Chapter on Matrix Operations
   - Cache-friendly algorithms

3. **Deep Learning Frameworks**:
   - PyTorch internals (ATen tensor library)
   - NumPy stride tricks

4. **Performance**:
   - "What Every Programmer Should Know About Memory" by Ulrich Drepper
   - BLAS library documentation (OpenBLAS, Intel MKL)

---

**Next Tutorial**: Building an Automatic Differentiation Engine (Autograd)

Now let's make our tensors track gradients automatically! 🚀
