# Library Architecture

## Overview

This document outlines the overall architecture and module organization for the C++ neural network library. The design emphasizes modularity, extensibility, and clear separation of concerns.

## High-Level Architecture

### Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                High-Level API Layer                      │
│  (Model, Sequential, Trainer, DataLoader)                │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Components Layer                          │
│  (Layers, Optimizers, Losses, Activations)               │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Core Computation Layer                    │
│  (Tensor, Operations, Autograd)                          │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Backend Layer                             │
│  (CPU/GPU Implementation, BLAS, Memory)                  │
└─────────────────────────────────────────────────────────┘
```

## Core Modules

### tensor Module

**Purpose**: Fundamental n-dimensional array abstraction

**Components**:

**Tensor Class**:

```
namespace nn {

template<typename T>
class Tensor {
public:
    // Construction
    Tensor()
    Tensor(std::vector<size_t> shape)
    Tensor(std::vector<size_t> shape, const T* data)
    
    // Shape and properties
    size_t ndim() const
    std::vector<size_t> shape() const
    size_t size() const
    bool requires_grad() const
    
    // Data access
    T* data()
    const T* data() const
    T& at(std::vector<size_t> indices)
    
    // Operations (return new tensors)
    Tensor operator+(const Tensor& other) const
    Tensor operator-(const Tensor& other) const
    Tensor operator*(const Tensor& other) const  // element-wise
    Tensor matmul(const Tensor& other) const
    Tensor transpose(int dim0, int dim1) const
    
    // In-place operations
    Tensor& add_(const Tensor& other)
    Tensor& mul_(T scalar)
    
    // Views and reshaping
    Tensor reshape(std::vector<size_t> new_shape) const
    Tensor view(std::vector<size_t> new_shape) const
    Tensor slice(/* args */) const
    
    // Reduction operations
    Tensor sum(int axis = -1) const
    Tensor mean(int axis = -1) const
    Tensor max(int axis = -1) const
    
    // Autograd support
    void backward(const Tensor& gradient = Tensor())
    Tensor grad() const
    void retain_grad()
    void zero_grad()
    
private:
    std::shared_ptr<TensorImpl> impl_
}

} // namespace nn
```

**TensorImpl** (Internal):

```
class TensorImpl {
    std::vector<size_t> shape_
    std::shared_ptr<Storage> storage_  // Actual data storage
    size_t offset_  // For views
    std::vector<size_t> strides_
    
    // Autograd
    std::shared_ptr<Tensor> grad_
    std::shared_ptr<Function> grad_fn_
    bool requires_grad_
}
```

**Storage** (Backend abstraction):

```
template<typename T>
class Storage {
public:
    virtual T* data() = 0
    virtual size_t size() const = 0
    virtual DeviceType device() const = 0
    
    virtual ~Storage() = default
}

class CPUStorage : public Storage<T> { /* ... */ }
class GPUStorage : public Storage<T> { /* ... */ }  // If GPU support
```

### ops Module

**Purpose**: Tensor operations with autograd support

**Structure**:

```
namespace nn::ops {

// Forward operations
Tensor add(const Tensor& a, const Tensor& b)
Tensor mul(const Tensor& a, const Tensor& b)
Tensor matmul(const Tensor& a, const Tensor& b)
Tensor conv2d(const Tensor& input, const Tensor& weight, /* params */)
Tensor relu(const Tensor& input)

// Each operation has corresponding backward function
// Managed by autograd system

} // namespace nn::ops
```

**Operation Implementation Pattern**:

```
class AddOp {
public:
    static Tensor forward(const Tensor& a, const Tensor& b) {
        // Perform addition
        Tensor result = /* actual computation */
        
        // Set up autograd
        if (a.requires_grad() || b.requires_grad()) {
            result.set_grad_fn(std::make_shared<AddBackward>(a, b))
        }
        
        return result
    }
}

class AddBackward : public Function {
public:
    std::vector<Tensor> backward(const Tensor& grad_output) override {
        // Gradient w.r.t. both inputs is just grad_output
        return {grad_output, grad_output}
    }
    
private:
    Tensor a_, b_  // Saved for backward
}
```

### autograd Module

**Purpose**: Automatic differentiation

**Components**:

**Function Base Class**:

```
class Function {
public:
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0
    virtual ~Function() = default
    
    std::vector<std::shared_ptr<Function>> next_functions_
}
```

**Computation Graph**:

```
class ComputationGraph {
public:
    void add_node(std::shared_ptr<Function> node)
    std::vector<std::shared_ptr<Function>> topological_sort()
    
    void backward(const Tensor& loss)
}
```

**Backward Pass**:

```
void backward(const Tensor& loss) {
    // Topological sort of computation graph
    auto sorted = topological_sort(loss)
    
    // Initialize gradient
    std::unordered_map<Function*, Tensor> gradients
    gradients[loss.grad_fn().get()] = ones_like(loss)
    
    // Reverse order traversal
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        auto node = *it
        auto grad_output = gradients[node.get()]
        
        // Compute gradients for inputs
        auto grad_inputs = node->backward(grad_output)
        
        // Accumulate to inputs
        for (size_t i = 0; i < grad_inputs.size(); ++i) {
            auto next = node->next_functions_[i]
            if (next) {
                gradients[next.get()] += grad_inputs[i]
            }
        }
    }
}
```

### layers Module

**Purpose**: Neural network layer implementations

**Organization**:

```
nn/
├── layers/
│   ├── layer.h          // Base layer interface
│   ├── dense.h          // Fully connected layer
│   ├── conv2d.h         // 2D convolution
│   ├── pooling.h        // Max/average pooling
│   ├── activation.h     // Activation layers
│   ├── dropout.h        // Dropout layer
│   ├── batch_norm.h     // Batch normalization
│   ├── recurrent.h      // RNN, LSTM, GRU
│   └── embedding.h      // Embedding layer
```

**Base Layer Interface**:

```
namespace nn {

class Layer {
public:
    virtual ~Layer() = default
    
    // Forward pass
    virtual Tensor forward(const Tensor& input) = 0
    
    // Get parameters
    virtual std::vector<Parameter*> parameters() {
        return {}
    }
    
    // Training mode control
    virtual void train() { training_ = true }
    virtual void eval() { training_ = false }
    bool is_training() const { return training_ }
    
    // Serialization
    virtual void save(std::ostream& stream) const = 0
    virtual void load(std::istream& stream) = 0
    
    // Metadata
    virtual std::string name() const = 0
    
protected:
    bool training_ = true
}

} // namespace nn
```

**Parameter Class**:

```
class Parameter {
public:
    Parameter(Tensor value, bool requires_grad = true)
    
    Tensor& value() { return value_ }
    const Tensor& value() const { return value_ }
    
    Tensor& grad() { return grad_ }
    void zero_grad() { grad_.zero_() }
    
    bool requires_grad() const { return requires_grad_ }
    
private:
    Tensor value_
    Tensor grad_
    bool requires_grad_
}
```

### nn Module

**Purpose**: High-level model abstractions

**Components**:

**Sequential**:

```
class Sequential : public Layer {
public:
    Sequential& add(std::unique_ptr<Layer> layer) {
        layers_.push_back(std::move(layer))
        return *this
    }
    
    template<typename LayerType, typename... Args>
    Sequential& add(Args&&... args) {
        return add(std::make_unique<LayerType>(std::forward<Args>(args)...))
    }
    
    Tensor forward(const Tensor& input) override {
        Tensor x = input
        for (auto& layer : layers_) {
            x = layer->forward(x)
        }
        return x
    }
    
    std::vector<Parameter*> parameters() override {
        std::vector<Parameter*> params
        for (auto& layer : layers_) {
            auto layer_params = layer->parameters()
            params.insert(params.end(), layer_params.begin(), layer_params.end())
        }
        return params
    }
    
private:
    std::vector<std::unique_ptr<Layer>> layers_
}
```

**Module** (For custom models):

```
class Module {
public:
    virtual Tensor forward(const Tensor& input) = 0
    
    // Register submodules
    template<typename T>
    void register_module(const std::string& name, std::shared_ptr<T> module)
    
    // Register parameters
    void register_parameter(const std::string& name, Parameter param)
    
    // Get all parameters recursively
    std::vector<Parameter*> parameters()
    
    void train() / void eval()
}
```

### optim Module

**Purpose**: Optimization algorithms

**Organization**:

```
nn/
├── optim/
│   ├── optimizer.h      // Base optimizer
│   ├── sgd.h            // SGD + momentum
│   ├── adam.h           // Adam optimizer
│   ├── adamw.h          // AdamW
│   ├── rmsprop.h        // RMSprop
│   └── scheduler.h      // Learning rate schedulers
```

**Optimizer Base**:

```
class Optimizer {
public:
    Optimizer(std::vector<Parameter*> parameters, float learning_rate)
        : parameters_(parameters), lr_(learning_rate) {}
    
    virtual void step() = 0
    
    void zero_grad() {
        for (auto* param : parameters_) {
            param->zero_grad()
        }
    }
    
    void set_learning_rate(float lr) { lr_ = lr }
    float learning_rate() const { return lr_ }
    
    virtual ~Optimizer() = default
    
protected:
    std::vector<Parameter*> parameters_
    float lr_
}
```

**State Management** (for Adam, etc.):

```
class StatefulOptimizer : public Optimizer {
protected:
    // State per parameter
    std::unordered_map<Parameter*, OptimizerState> state_
    
    struct OptimizerState {
        Tensor momentum  // For momentum/Adam m
        Tensor velocity  // For Adam v
        size_t step      // Step counter
    }
}
```

### loss Module

**Purpose**: Loss function implementations

**Organization**:

```
nn/
├── loss/
│   ├── loss.h              // Base loss interface
│   ├── mse_loss.h          // Mean squared error
│   ├── cross_entropy.h     // Cross entropy
│   ├── bce_loss.h          // Binary cross entropy
│   └── huber_loss.h        // Huber loss
```

**Loss Interface**:

```
class Loss {
public:
    virtual Tensor forward(const Tensor& predictions, const Tensor& targets) = 0
    virtual ~Loss() = default
}

// Gradient computed via autograd automatically
```

### data Module

**Purpose**: Data loading and processing

**Components**:

**Dataset**:

```
template<typename SampleType>
class Dataset {
public:
    virtual size_t size() const = 0
    virtual SampleType operator[](size_t index) const = 0
    virtual ~Dataset() = default
}
```

**DataLoader**:

```
template<typename SampleType>
class DataLoader {
public:
    DataLoader(
        std::shared_ptr<Dataset<SampleType>> dataset,
        size_t batch_size,
        bool shuffle = true,
        size_t num_workers = 0
    )
    
    class Iterator {
        // Batch iteration
    }
    
    Iterator begin()
    Iterator end()
    
private:
    std::shared_ptr<Dataset<SampleType>> dataset_
    size_t batch_size_
    bool shuffle_
    std::vector<size_t> indices_
}
```

### utils Module

**Purpose**: Utility functions

**Components**:

```
nn/
├── utils/
│   ├── initializers.h    // Weight initialization
│   ├── metrics.h         // Accuracy, precision, etc.
│   ├── io.h              // Save/load utilities
│   └── random.h          // Random number generation
```

**Initializers**:

```
namespace nn::init {

void xavier_uniform(Tensor& tensor, float gain = 1.0)
void xavier_normal(Tensor& tensor, float gain = 1.0)
void he_uniform(Tensor& tensor)
void he_normal(Tensor& tensor)
void zeros(Tensor& tensor)
void ones(Tensor& tensor)
void constant(Tensor& tensor, float value)

} // namespace nn::init
```

## Backend Abstraction

### Device Management

**Device Type**:

```
enum class DeviceType {
    CPU,
    CUDA  // If GPU support
}

class Device {
public:
    Device(DeviceType type, int index = 0)
    
    DeviceType type() const
    int index() const
    
    static Device cpu()
    static Device cuda(int index = 0)
    
private:
    DeviceType type_
    int index_
}
```

**Tensor-Device Interaction**:

```
Tensor tensor = /* ... */
tensor.to(Device::cuda())  // Move to GPU
tensor.to(Device::cpu())   // Move to CPU
```

### Backend Dispatch

**Operation Dispatch**:

```
class DispatchKey {
    DeviceType device
    DataType dtype
}

// Operation registry
template<typename Op>
struct OperationRegistry {
    static auto& get(DispatchKey key) {
        // Return appropriate implementation
    }
}

// Example
Tensor add(const Tensor& a, const Tensor& b) {
    auto key = DispatchKey{a.device(), a.dtype()}
    return OperationRegistry<AddOp>::get(key)(a, b)
}
```

### BLAS Integration

**Matrix Operations**:

```
namespace nn::backend {

// Abstract interface
class BLASInterface {
public:
    virtual void gemm(/* params */) = 0  // Matrix multiply
    virtual void gemv(/* params */) = 0  // Matrix-vector
    virtual void axpy(/* params */) = 0  // y = a*x + y
}

// Implementations
class OpenBLAS : public BLASInterface { /* ... */ }
class MKL : public BLASInterface { /* ... */ }
class cuBLAS : public BLASInterface { /* GPU */ }

// Singleton access
BLASInterface& get_blas()

} // namespace nn::backend
```

## Memory Management Strategy

### Tensor Memory

**Reference Counting**:

- `std::shared_ptr` for storage
- Shallow copies share data
- Copy-on-write for mutations (optional)

**Memory Pool** (Optional optimization):

```
class MemoryPool {
public:
    void* allocate(size_t bytes, Device device)
    void deallocate(void* ptr, Device device)
    
    void clear_cache()
    
private:
    std::unordered_map<Device, std::vector<Block>> free_blocks_
}
```

### Gradient Memory

**Backward Pass**:

- Allocate gradient tensors lazily
- Clear after optimizer step
- Option to retain for debugging

## Thread Safety

### Concurrent Operations

**Read-Only Operations** (Thread-Safe):

- Forward pass (inference)
- Parameter access (const)

**Write Operations** (Not Thread-Safe):

- Backward pass
- Parameter updates
- Requires external synchronization

**Parallel Data Loading**:

- Multiple worker threads
- Thread-safe queue
- Synchronization at batch boundaries

## Plugin Architecture

### Extensibility Points

**Custom Layers**:

```
class CustomLayer : public Layer {
    Tensor forward(const Tensor& input) override {
        // Custom logic
    }
}

// Use in Sequential
model.add(std::make_unique<CustomLayer>())
```

**Custom Operations**:

```
// Register new operation
REGISTER_OPERATION(CustomOp, DeviceType::CPU, float)
```

**Custom Optimizers**:

```
class CustomOptimizer : public Optimizer {
    void step() override {
        // Custom update logic
    }
}
```

## Build System

### CMake Structure

```
CMakeLists.txt
nn/
├── CMakeLists.txt
├── tensor/
│   └── CMakeLists.txt
├── ops/
│   └── CMakeLists.txt
├── layers/
│   └── CMakeLists.txt
└── ...

# Main CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(NeuralNetwork CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(NN_BUILD_TESTS "Build tests" ON)
option(NN_BUILD_EXAMPLES "Build examples" ON)
option(NN_ENABLE_GPU "Enable GPU support" OFF)

# Find dependencies
find_package(OpenBLAS)  # Or MKL

# Add subdirectories
add_subdirectory(nn)
if(NN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

### Dependency Management

**Required**:

- C++17 compiler
- BLAS library (OpenBLAS, MKL, or Accelerate)

**Optional**:

- CUDA (for GPU support)
- Threading library

## Testing Strategy

### Unit Tests

**Per-Module Tests**:

```
tests/
├── tensor_test.cpp
├── ops_test.cpp
├── layer_test.cpp
├── optimizer_test.cpp
└── integration_test.cpp
```

**Test Framework**: Google Test or Catch2

**Example Test**:

```
TEST(TensorTest, Addition) {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6})
    Tensor b({2, 3}, {1, 1, 1, 1, 1, 1})
    
    Tensor c = a + b
    
    EXPECT_EQ(c.shape(), std::vector<size_t>{2, 3})
    EXPECT_FLOAT_EQ(c.at({0, 0}), 2.0f)
}
```

### Gradient Checking

**Numerical Gradient Verification**:

```
bool gradient_check(
    std::function<Tensor(Tensor)> fn,
    Tensor input,
    float epsilon = 1e-5
) {
    // Compute gradient via backprop
    // Compute numerical gradient
    // Compare
}
```

## Documentation

### API Documentation

**Doxygen Format**:

```cpp
/**
 * @brief Performs matrix multiplication
 * 
 * @param a First tensor (M x K)
 * @param b Second tensor (K x N)
 * @return Tensor Result tensor (M x N)
 * 
 * @throws ShapeError if dimensions incompatible
 */
Tensor matmul(const Tensor& a, const Tensor& b)
```

### User Guide

**Documentation Structure**:

```
docs/
├── getting_started.md
├── tutorials/
│   ├── building_first_network.md
│   ├── custom_layers.md
│   └── training_tips.md
├── api/
│   ├── tensor.md
│   ├── layers.md
│   └── optimizers.md
└── examples/
    ├── mnist.cpp
    ├── cifar10.cpp
    └── custom_model.cpp
```

## Directory Structure

```
neural-network/
├── include/
│   └── nn/
│       ├── tensor.h
│       ├── layer.h
│       ├── sequential.h
│       ├── optimizer.h
│       └── ...
├── src/
│   ├── tensor/
│   │   ├── tensor.cpp
│   │   └── ops.cpp
│   ├── layers/
│   │   ├── dense.cpp
│   │   ├── conv2d.cpp
│   │   └── ...
│   ├── optim/
│   └── ...
├── tests/
├── examples/
├── docs/
├── CMakeLists.txt
└── README.md
```

## Summary

### Architectural Principles

1. **Modularity**: Clear module boundaries
2. **Extensibility**: Plugin architecture for custom components
3. **Performance**: Backend abstraction for optimization
4. **Safety**: RAII, smart pointers, type safety
5. **Testability**: Comprehensive unit tests

### Key Design Decisions

- Tensor as core abstraction
- Automatic differentiation via computation graph
- Layer-based composition
- Backend dispatch for CPU/GPU
- BLAS integration for performance
- Smart pointer-based memory management

### Implementation Priorities

**Phase 1** (Core):

- Tensor implementation
- Basic operations
- Autograd system
- Dense layers

**Phase 2** (Essentials):

- More layer types (Conv, etc.)
- Optimizers (SGD, Adam)
- Loss functions
- Data loading

**Phase 3** (Advanced):

- GPU support
- Additional architectures (RNN, etc.)
- Advanced optimizations
- Distributed training

---

**Previous**: [API Design Principles](./10-api-design-principles.md)  
**Next**: [Performance Considerations](./12-performance-considerations.md)
