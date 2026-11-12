# Mathematical Foundations

## Overview

This document covers the essential mathematical concepts required to understand and implement neural networks. These foundations are critical for both theoretical understanding and practical implementation.

## Linear Algebra Essentials

### Scalars, Vectors, and Matrices

#### Scalar

A single numerical value, denoted by lowercase letters: `a`, `b`, `c`

**Properties**:

- Dimensionality: 0D
- Example: `x = 5`

#### Vector

An ordered array of numbers, denoted by lowercase bold letters: **v**, **x**

**Properties**:

- Dimensionality: 1D
- Length/dimension: n
- Notation: **v** ∈ ℝⁿ

**Example**:

```
v = [v₁, v₂, v₃, ..., vₙ]ᵀ
```

**Column vs Row Vectors**:

- Column vector: **v** = [v₁, v₂, ..., vₙ]ᵀ (default)
- Row vector: **v**ᵀ = [v₁, v₂, ..., vₙ]

#### Matrix

A 2D array of numbers, denoted by uppercase bold letters: **A**, **B**, **W**

**Properties**:

- Dimensionality: 2D
- Shape: m × n (m rows, n columns)
- Notation: **A** ∈ ℝᵐˣⁿ

**Example**:

```
A = [a₁₁  a₁₂  ...  a₁ₙ]
    [a₂₁  a₂₂  ...  a₂ₙ]
    [  ⋮    ⋮   ⋱    ⋮ ]
    [aₘ₁  aₘ₂  ...  aₘₙ]
```

#### Tensor

A multi-dimensional array, generalization of matrices

**Properties**:

- Dimensionality: nD (n > 2)
- Example: 3D tensor for RGB images: height × width × channels

### Vector Operations

#### Vector Addition

**Definition**: Element-wise addition of vectors of the same dimension

**Mathematical Form**:

```
c = a + b
cᵢ = aᵢ + bᵢ, for i = 1, 2, ..., n
```

**Properties**:

- Commutative: **a** + **b** = **b** + **a**
- Associative: (**a** + **b**) + **c** = **a** + (**b** + **c**)
- Identity: **a** + **0** = **a**

**Requirements**:

- Both vectors must have the same dimension

#### Scalar Multiplication

**Definition**: Multiplication of a vector by a scalar

**Mathematical Form**:

```
b = αa
bᵢ = αaᵢ, for i = 1, 2, ..., n
```

**Properties**:

- Distributive: α(**a** + **b**) = α**a** + α**b**
- Associative: (αβ)**a** = α(β**a**)

#### Dot Product (Inner Product)

**Definition**: Scalar result from multiplying corresponding elements and summing

**Mathematical Form**:

```
c = a · b = aᵀb = Σ(aᵢbᵢ) from i=1 to n
```

**Properties**:

- Commutative: **a** · **b** = **b** · **a**
- Distributive: **a** · (**b** + **c**) = **a** · **b** + **a** · **c**
- Result: Scalar value

**Geometric Interpretation**:

```
a · b = ||a|| ||b|| cos(θ)
```

where θ is the angle between vectors

**Requirements**:

- Both vectors must have the same dimension

### Matrix Operations

#### Matrix Addition

**Definition**: Element-wise addition of matrices

**Mathematical Form**:

```
C = A + B
Cᵢⱼ = Aᵢⱼ + Bᵢⱼ
```

**Requirements**:

- Matrices must have the same shape (m × n)

**Computational Complexity**: O(mn)

#### Scalar-Matrix Multiplication

**Definition**: Each element multiplied by scalar

**Mathematical Form**:

```
B = αA
Bᵢⱼ = αAᵢⱼ
```

**Computational Complexity**: O(mn)

#### Matrix Multiplication

**Definition**: Core operation in neural networks

**Mathematical Form**:

```
C = AB
where A ∈ ℝᵐˣⁿ, B ∈ ℝⁿˣᵖ, C ∈ ℝᵐˣᵖ

Cᵢⱼ = Σ(Aᵢₖ Bₖⱼ) from k=1 to n
```

**Requirements**:

- Number of columns in A must equal number of rows in B
- If A is (m × n) and B is (n × p), result C is (m × p)

**Properties**:

- NOT commutative: AB ≠ BA (in general)
- Associative: (AB)C = A(BC)
- Distributive: A(B + C) = AB + AC

**Computational Complexity**: O(mnp) (naive algorithm)

**Optimization Note**: Highly optimized implementations (e.g., BLAS libraries) use Strassen's algorithm or cache-friendly blocked algorithms

#### Element-wise Multiplication (Hadamard Product)

**Definition**: Element-wise multiplication, denoted by ⊙

**Mathematical Form**:

```
C = A ⊙ B
Cᵢⱼ = Aᵢⱼ × Bᵢⱼ
```

**Requirements**:

- Matrices must have the same shape

**Use in Neural Networks**: Applied in activation functions and gradient computations

#### Matrix Transpose

**Definition**: Flip matrix over its diagonal

**Mathematical Form**:

```
If A ∈ ℝᵐˣⁿ, then Aᵀ ∈ ℝⁿˣᵐ
(Aᵀ)ᵢⱼ = Aⱼᵢ
```

**Properties**:

- (Aᵀ)ᵀ = A
- (AB)ᵀ = BᵀAᵀ
- (A + B)ᵀ = Aᵀ + Bᵀ

### Matrix Properties

#### Identity Matrix

**Definition**: Square matrix with 1s on diagonal, 0s elsewhere

**Mathematical Form**:

```
I = [1  0  ...  0]
    [0  1  ...  0]
    [⋮  ⋮   ⋱   ⋮]
    [0  0  ...  1]

Iᵢⱼ = { 1 if i = j
      { 0 if i ≠ j
```

**Property**: AI = IA = A

#### Inverse Matrix

**Definition**: Matrix A⁻¹ such that AA⁻¹ = A⁻¹A = I

**Requirements**:

- Matrix must be square (n × n)
- Matrix must be non-singular (determinant ≠ 0)

**Use in Neural Networks**: Used in some optimization methods (though often avoided due to computational cost)

## Calculus for Machine Learning

### Derivatives

#### Derivative Definition

**Definition**: Rate of change of a function

**Mathematical Form**:

```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

**Notation Variants**:

- f'(x)
- df/dx
- ∂f/∂x (partial derivative)

#### Common Derivatives

**Power Rule**:

```
f(x) = xⁿ
f'(x) = nxⁿ⁻¹
```

**Exponential**:

```
f(x) = eˣ
f'(x) = eˣ
```

**Logarithm**:

```
f(x) = ln(x)
f'(x) = 1/x
```

**Trigonometric**:

```
f(x) = sin(x),  f'(x) = cos(x)
f(x) = cos(x),  f'(x) = -sin(x)
```

### Partial Derivatives

**Definition**: Derivative with respect to one variable, holding others constant

**Mathematical Form**:

```
For f(x, y):
∂f/∂x = rate of change with respect to x (y held constant)
∂f/∂y = rate of change with respect to y (x held constant)
```

**Example**:

```
f(x, y) = x²y + y³
∂f/∂x = 2xy
∂f/∂y = x² + 3y²
```

### Gradient

**Definition**: Vector of all partial derivatives

**Mathematical Form**:

```
For f: ℝⁿ → ℝ, gradient is:

∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

**Properties**:

- Points in direction of steepest ascent
- Magnitude indicates rate of change
- Perpendicular to level curves/surfaces

**Use in Neural Networks**: Gradient descent optimization

### Chain Rule

**Single Variable Form**:

```
If h(x) = f(g(x)), then:
h'(x) = f'(g(x)) · g'(x)
```

**Multivariate Form**:

```
If z = f(y) and y = g(x), then:
∂z/∂x = (∂z/∂y) · (∂y/∂x)
```

**General Chain Rule**:

```
For z = f(y₁, y₂, ..., yₙ) where each yᵢ = gᵢ(x):

∂z/∂x = Σ(∂z/∂yᵢ · ∂yᵢ/∂x) from i=1 to n
```

**Critical Importance**: Foundation of backpropagation algorithm

### Jacobian Matrix

**Definition**: Matrix of all first-order partial derivatives for vector-valued functions

**Mathematical Form**:

```
For f: ℝⁿ → ℝᵐ, Jacobian is m × n matrix:

J = [∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ]
    [∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ]
    [   ⋮        ⋮      ⋱      ⋮   ]
    [∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ]
```

**Use in Neural Networks**: Computing gradients for vector-valued layer outputs

### Hessian Matrix

**Definition**: Matrix of second-order partial derivatives

**Mathematical Form**:

```
For f: ℝⁿ → ℝ, Hessian is n × n matrix:

H = [∂²f/∂x₁∂x₁  ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ]
    [∂²f/∂x₂∂x₁  ∂²f/∂x₂∂x₂  ...  ∂²f/∂x₂∂xₙ]
    [     ⋮            ⋮       ⋱        ⋮     ]
    [∂²f/∂xₙ∂x₁  ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ∂xₙ]
```

**Properties**:

- Symmetric if f is twice continuously differentiable
- Used in second-order optimization methods

**Use in Neural Networks**: Advanced optimization methods (Newton's method, L-BFGS)

## Gradient Computation

### Computational Graph Perspective

**Concept**: Neural networks can be viewed as computational graphs where:

- Nodes represent operations or variables
- Edges represent data flow
- Gradients flow backward through the graph

### Forward Mode vs Reverse Mode

#### Forward Mode Differentiation

**Process**:

1. Compute function values forward
2. Simultaneously compute derivatives forward

**Efficiency**:

- Efficient when: few inputs, many outputs
- Complexity: O(n) passes for n inputs

#### Reverse Mode Differentiation (Backpropagation)

**Process**:

1. Forward pass: compute all intermediate values
2. Backward pass: compute gradients by chain rule

**Efficiency**:

- Efficient when: many inputs, few outputs
- Complexity: O(m) passes for m outputs
- **Preferred for neural networks** (typically 1 output: loss)

### Vector-Jacobian Product

**Concept**: Core operation in backpropagation

**Mathematical Form**:

```
Given:
- Forward function: y = f(x)
- Gradient from upstream: ∂L/∂y

Compute:
- Gradient for x: ∂L/∂x = (∂L/∂y) · (∂y/∂x)
```

**Efficiency**: Computing vector-Jacobian product is more efficient than computing full Jacobian

## Numerical Considerations

### Numerical Stability

#### Issues to Consider

1. **Overflow**: Values too large to represent
2. **Underflow**: Values too small, rounded to zero
3. **Catastrophic Cancellation**: Loss of precision in subtraction

#### Examples in Neural Networks

- **Softmax overflow**: exp(x) for large x
- **Log underflow**: log(0) or log(very small number)
- **Gradient vanishing**: Repeated multiplication of small numbers

### Precision Types

**Single Precision (float32)**:

- Range: ≈ 10⁻³⁸ to 10³⁸
- Precision: ≈ 7 decimal digits
- Most common in deep learning

**Double Precision (float64)**:

- Range: ≈ 10⁻³⁰⁸ to 10³⁰⁸
- Precision: ≈ 15 decimal digits
- Used for high-precision requirements

**Half Precision (float16)**:

- Range: ≈ 10⁻⁸ to 10⁴
- Precision: ≈ 3 decimal digits
- Used for memory/speed optimization

## Summary

### Key Takeaways

1. **Matrix multiplication** is the core operation in neural networks
2. **Chain rule** enables backpropagation
3. **Gradient** tells us how to update parameters
4. **Computational graphs** provide structure for automatic differentiation
5. **Numerical stability** must be considered in all operations

### Implementation Priorities

1. Efficient matrix operations (consider BLAS libraries)
2. Automatic differentiation system
3. Numerically stable implementations of common operations
4. Support for different precision types
5. Clear abstractions for tensors and operations

---

**Next**: [Neural Network Fundamentals](./02-neural-network-fundamentals.md)
