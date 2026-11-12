# Performance Considerations

## Overview

This document covers performance analysis, optimization strategies, and benchmarking criteria for the neural network library. Understanding computational complexity and optimization opportunities is crucial for building an efficient implementation.

## Computational Complexity Analysis

### Matrix Operations

**Matrix Multiplication** (core operation):

```
C = A × B
where A ∈ ℝᵐˣⁿ, B ∈ ℝⁿˣᵖ, C ∈ ℝᵐˣᵖ

Naive Algorithm:
Time: O(mnp)
Space: O(mp) for output

Strassen's Algorithm:
Time: O(n^2.807) for square matrices
Practical only for very large matrices
```

**Element-wise Operations**:

```
Time: O(n) where n is total elements
Space: O(n) for output (can be in-place for some ops)

Examples:
- Addition: a + b
- ReLU: max(0, x)
- Sigmoid: 1/(1 + exp(-x))
```

**Reductions**:

```
Sum, Mean, Max, etc. over axes

Time: O(n)
Space: O(output_size)
```

### Layer Complexity

**Dense Layer**:

```
Forward:
  y = Wx + b
  W ∈ ℝᵐˣⁿ, x ∈ ℝⁿ, b ∈ ℝᵐ
  
Time: O(mn) - matrix-vector multiply
Space: O(m) - output

For batch of size B:
Time: O(Bmn)
Space: O(Bm)

Backward:
  ∂L/∂W = ∂L/∂y · x^T  (outer product)
  ∂L/∂x = W^T · ∂L/∂y
  
Time: O(mn) per example, O(Bmn) for batch
```

**Convolutional Layer**:

```
Input: H × W × C_in
Filter: K × K × C_in × C_out
Output: H' × W' × C_out

Forward:
Time: O(H' · W' · K² · C_in · C_out)
Space: O(H' · W' · C_out)

For batch B:
Time: O(B · H' · W' · K² · C_in · C_out)

Backward:
Similar complexity for both weight and input gradients
```

**RNN/LSTM**:

```
Hidden size: h
Input size: d
Sequence length: T

RNN Forward:
Time per step: O(h² + hd)
Total: O(T(h² + hd))

LSTM Forward:
Time per step: O(4(h² + hd))  // 4 gates
Total: O(4T(h² + hd))

Backward:
Same complexity as forward (BPTT)
```

### Full Network Complexity

**Feedforward Network**:

```
Layers: n₀ → n₁ → n₂ → ... → nₗ

Forward Pass:
Time: O(Σᵢ nᵢ · nᵢ₊₁)
Space: O(Σᵢ nᵢ) for activations (caching)

Backward Pass:
Same complexity as forward

Parameters:
Total: O(Σᵢ nᵢ · nᵢ₊₁)
```

### Batch Processing Impact

**Batch Size Effect**:

```
Single example: Time per update = t
Batch of B: Time per update ≈ t/k where k < B

Why k < B?
- Parallelization (GPU)
- Vectorization (SIMD)
- Cache efficiency
- Amortized overhead

Typical speedup: 5-50x for moderate batch sizes
```

## Memory Analysis

### Memory Requirements

**Model Storage**:

```
Parameters: Σ (weights + biases)

Example:
Dense 784 → 256 → 128 → 10

Weights: (784·256 + 256·128 + 128·10) = 234,752 floats
Biases: (256 + 128 + 10) = 394 floats
Total: ~235K floats × 4 bytes = ~940 KB

Note: Modern models have millions to billions of parameters
```

**Activation Storage** (for backprop):

```
Must store all intermediate activations

For batch size B:
Space: O(B · Σᵢ nᵢ)

This is often the bottleneck for large batches!
```

**Optimizer State**:

```
SGD: No additional storage
SGD + Momentum: 1× parameters (velocity)
Adam: 2× parameters (m and v)
AdamW: 2× parameters

Example: Adam with 10M parameters:
- Parameters: 10M floats = 40 MB
- Gradients: 10M floats = 40 MB
- m state: 10M floats = 40 MB
- v state: 10M floats = 40 MB
Total: 160 MB
```

**Gradient Storage**:

```
Same size as parameters

For large models, can use gradient accumulation
to trade time for memory
```

### Memory Optimization Strategies

**Gradient Checkpointing**:

```
Trade compute for memory

Instead of storing all activations:
1. Store only subset of activations
2. Recompute others during backward pass

Memory reduction: O(√L) for L-layer network
Compute increase: ~33% (one additional forward pass)
```

**In-Place Operations**:

```
Where possible, modify tensors in-place

Example:
a += b  // In-place, no new allocation
vs
a = a + b  // May allocate new tensor

ReLU can be in-place:
x = max(0, x)  // Overwrites x
```

**Memory Pooling**:

```
Reuse deallocated memory

Instead of:
- Allocate for each tensor
- Deallocate immediately after use

Do:
- Maintain pool of free blocks
- Reuse blocks of similar size
- Only deallocate on request or destruction
```

**Mixed Precision**:

```
Use FP16 where possible, FP32 where needed

Memory reduction: 2× for activations
Performance: Often 2× faster on modern GPUs

Careful: Loss of precision can affect training
```

## Parallelization Strategies

### Data Parallelism

**Mini-Batch Parallelism** (GPU/SIMD):

```
Process multiple examples simultaneously

GPU: Thousands of threads
CPU SIMD: 4-8-16 elements simultaneously

Effective for: matrix operations on batches
```

**Multi-GPU Data Parallelism**:

```
Split batch across GPUs

For K GPUs, batch size B:
- GPU 1: Examples 1..B/K
- GPU 2: Examples B/K+1..2B/K
- ...

Synchronize gradients at end of batch
```

### Model Parallelism

**Layer Parallelism**:

```
Different layers on different devices

Sequential execution:
GPU 1: Layer 1-5
GPU 2: Layer 6-10
...

Pipeline parallelism: Further optimization
```

**Tensor Parallelism**:

```
Split single layer across devices

Example: Dense layer
W ∈ ℝᵐˣⁿ split into W₁, W₂

GPU 1: y₁ = W₁x
GPU 2: y₂ = W₂x
Combine: y = [y₁; y₂]
```

### CPU Parallelization

**Thread-Level Parallelism**:

```
Use multiple CPU threads

OpenMP for loop parallelization:
#pragma omp parallel for
for (size_t i = 0; i < n; ++i) {
    // Parallel work
}

Good for: Large matrix operations, batch processing
```

**SIMD Vectorization**:

```
Process multiple elements per instruction

Example (4-wide SIMD):
for (size_t i = 0; i < n; i += 4) {
    // Process 4 elements simultaneously
    __m128 a = _mm_load_ps(&data[i])
    // SIMD operations
}

Auto-vectorization: Compiler can often do this automatically
```

## Hardware Acceleration

### BLAS Libraries

**Why BLAS**:

- Highly optimized matrix operations
- Hardware-specific tuning
- Orders of magnitude faster than naive implementation

**Popular Libraries**:

**OpenBLAS**:

```
Open source, good performance
Multi-threaded
Supports many architectures
```

**Intel MKL**:

```
Optimized for Intel CPUs
Commercial (free for some uses)
Excellent performance on Intel hardware
```

**Apple Accelerate**:

```
Optimized for Apple Silicon
Built into macOS
Excellent performance on M1/M2
```

**NVIDIA cuBLAS**:

```
GPU-accelerated
Essential for CUDA
Extremely fast for large matrices
```

**Matrix Multiply Performance**:

```
Naive C++: ~5 GFLOPS
Optimized (BLAS): 100-1000 GFLOPS (CPU)
cuBLAS (GPU): 5000-15000 GFLOPS

3-4 orders of magnitude difference!
```

### GPU Acceleration (CUDA)

**When to Use GPU**:

```
Pros:
- Massive parallelism (thousands of cores)
- High memory bandwidth
- Essential for large models/datasets

Cons:
- GPU memory limited (4-80 GB typical)
- PCIe transfer overhead (CPU ↔ GPU)
- More complex code
```

**GPU Performance Considerations**:

```
Kernel Launch Overhead:
- ~5-20 microseconds per kernel
- Batch operations to amortize

Memory Transfer:
- PCIe bandwidth: ~16 GB/s (PCIe 3.0)
- GPU memory bandwidth: ~500-900 GB/s
- Keep data on GPU when possible

Occupancy:
- Fill GPU cores effectively
- Avoid thread divergence
- Coalesced memory access
```

**CUDA Libraries**:

```
cuBLAS: Matrix operations
cuDNN: Deep learning primitives
Thrust: Parallel algorithms
```

## Optimization Techniques

### Loop Optimizations

**Cache Blocking** (Tiling):

```
Matrix multiply: Improve cache locality

Instead of:
for i:
  for j:
    for k:
      C[i][j] += A[i][k] * B[k][j]

Do (blocked):
for ii in range(0, M, BLOCK):
  for jj in range(0, N, BLOCK):
    for kk in range(0, K, BLOCK):
      // Small block multiplication
      // Fits in cache
```

**Loop Unrolling**:

```
Reduce loop overhead

Instead of:
for (int i = 0; i < n; ++i)
    c[i] = a[i] + b[i]

Do:
for (int i = 0; i < n; i += 4) {
    c[i]   = a[i]   + b[i]
    c[i+1] = a[i+1] + b[i+1]
    c[i+2] = a[i+2] + b[i+2]
    c[i+3] = a[i+3] + b[i+3]
}
// Handle remainder
```

### Memory Access Patterns

**Spatial Locality**:

```
Access contiguous memory

Good: Sequential access
for (int i = 0; i < n; ++i)
    sum += array[i]

Bad: Strided access
for (int i = 0; i < n; i += stride)
    sum += array[i]
```

**Temporal Locality**:

```
Reuse recently accessed data

Matrix multiply: Block algorithms reuse data
```

**Memory Alignment**:

```
Align data to cache line boundaries (64 bytes typical)

alignas(64) float data[N]

Benefits:
- Faster loads/stores
- Required for some SIMD operations
```

### Kernel Fusion

**Combine Operations**:

```
Instead of:
x = relu(x)  // One kernel
x = dropout(x)  // Another kernel

Do:
x = dropout_relu(x)  // Fused kernel

Benefits:
- Fewer memory reads/writes
- Reduced kernel launch overhead
- Better cache usage
```

**Examples**:

```
- Batch normalization + activation
- Conv + batch norm + activation
- Multiple element-wise operations
```

### Computation Reordering

**Associativity Exploitation**:

```
For associative operations, reorder for efficiency

Vector dot product:
Standard: ((a₁b₁ + a₂b₂) + a₃b₃) + ...
Blocked: (a₁b₁ + a₂b₂) + (a₃b₃ + a₄b₄) + ...

Blocked allows better pipelining
```

**Strength Reduction**:

```
Replace expensive operations

Expensive: Division, exp, log, sqrt
Cheap: Multiply, add, bitwise ops

Example:
x / 2.0 → x * 0.5
```

## Operation-Specific Optimizations

### Matrix Multiplication

**Algorithm Choice**:

```
Small matrices (< 100): Naive O(n³)
Medium matrices: Blocked algorithm
Large matrices: BLAS library (always)

Never implement matrix multiply yourself for production!
```

**Memory Layout**:

```
Row-major vs Column-major

C (row-major): A[i][j] → A[i*cols + j]
Fortran (column-major): A[i][j] → A[i + j*rows]

BLAS expects column-major!
Transpose if needed, but note cost
```

### Convolution

**Algorithms**:

**Direct Convolution**:

```
Naive implementation
Good for: Small kernels
```

**Im2Col + GEMM**:

```
Transform convolution into matrix multiply
Use optimized GEMM (BLAS)
Memory overhead, but very fast
Most common in practice
```

**FFT-based Convolution**:

```
Use Fast Fourier Transform
Good for: Large kernels
Complexity: O(n log n) vs O(n²)
```

**Winograd**:

```
Minimal filtering
Reduces multiplications
Complex to implement, but used in some libraries
```

### Activation Functions

**Vectorization**:

```
Process multiple elements simultaneously

ReLU: Perfect for SIMD
max(0, x) for 4-16 elements at once

Sigmoid: Can approximate or use lookup table
```

**Approximations**:

```
Fast approximations for expensive functions

Fast sigmoid:
σ(x) ≈ x / (1 + |x|)  // Rough approximation
σ(x) ≈ accurate polynomial approximation

Fast tanh, exp, etc.
```

## Profiling and Benchmarking

### Profiling Tools

**CPU Profiling**:

```
- gprof (GCC)
- perf (Linux)
- Instruments (macOS)
- VTune (Intel)
- Tracy Profiler
```

**GPU Profiling**:

```
- NVIDIA Nsight
- nvprof
- cuda-memcheck
```

### Metrics to Track

**Performance Metrics**:

```
Throughput: Examples/second or FLOPS
Latency: Time per forward/backward pass
Memory: Peak usage, allocation rate
GPU Utilization: Percentage of time busy
```

**Bottleneck Analysis**:

```
CPU-bound: High CPU usage, low GPU usage
Memory-bound: Low compute utilization, high memory bandwidth
GPU-bound: High GPU usage, low CPU usage
I/O-bound: Waiting for data loading
```

### Benchmarking Methodology

**Micro-Benchmarks**:

```
Individual operations:
- Matrix multiply (various sizes)
- Convolution (various configurations)
- Activation functions
- Full layer forward/backward
```

**Macro-Benchmarks**:

```
Full models:
- ResNet-50 training (standardized)
- BERT training
- Custom models

Metrics:
- Time per epoch
- Images/second or tokens/second
- Memory usage
```

**Comparison Baseline**:

```
Compare against established libraries:
- PyTorch
- TensorFlow
- JAX

Goal: Within 1.5-2× of optimized libraries
(Matching exactly is very difficult without
 years of optimization)
```

## Performance Targets

### Realistic Expectations

**Dense Layer**:

```
Forward pass should be dominated by BLAS GEMM
Target: >80% of BLAS peak performance

Backward: Similar to forward
```

**Convolution**:

```
Im2Col + GEMM approach:
Target: >70% of BLAS peak (overhead for im2col)

Well-optimized libraries (cuDNN): 90%+
```

**Full Network**:

```
Training throughput:
- Small networks: 1000-10000 examples/sec
- Medium networks: 100-1000 examples/sec
- Large networks: 1-100 examples/sec

(Depends heavily on network size and hardware)
```

### Optimization Priorities

**Priority Order**:

1. **Use BLAS library** (10-100x speedup)
2. **Batch operations** (5-50x speedup)
3. **GPU acceleration** (5-50x speedup if applicable)
4. **Memory efficiency** (Enables larger models/batches)
5. **Cache optimization** (1.5-3x speedup)
6. **SIMD vectorization** (2-4x speedup)
7. **Algorithmic improvements** (Problem-specific)

## Summary

### Key Takeaways

1. **BLAS is essential**: Don't implement matrix ops yourself
2. **Batch processing**: Crucial for GPU/SIMD efficiency
3. **Memory is often bottleneck**: Cache awareness and pooling important
4. **Profile before optimizing**: Measure, don't guess
5. **GPU when appropriate**: Large models and datasets

### Performance Checklist

**Must Have**:

- BLAS library integration
- Efficient matrix layouts
- Batch processing support
- Memory pooling

**Should Have**:

- In-place operations where possible
- Kernel fusion (common combinations)
- SIMD-friendly code
- Cache-aware algorithms

**Nice to Have**:

- GPU support
- Gradient checkpointing
- Mixed precision training
- Custom GPU kernels

### Implementation Strategy

**Phase 1**: Correctness

- Get it working
- Use simple, clear algorithms
- Comprehensive testing

**Phase 2**: Basic Optimization

- Integrate BLAS
- Batch processing
- Basic memory management

**Phase 3**: Advanced Optimization

- Profiling and bottleneck identification
- Cache optimization
- SIMD vectorization
- GPU support (if needed)

---

**Previous**: [Library Architecture](./11-library-architecture.md)  
**Back to Index**: [README](./index.md)
