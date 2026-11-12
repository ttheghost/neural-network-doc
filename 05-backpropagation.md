# Backpropagation Algorithm

## Overview

Backpropagation is the fundamental algorithm for training neural networks. It efficiently computes gradients of the loss function with respect to all network parameters using the chain rule of calculus. This document provides a complete mathematical specification of the algorithm.

## Core Concept

### The Problem

**Given**:

- Network with parameters θ = {W⁽¹⁾, b⁽¹⁾, W⁽²⁾, b⁽²⁾, ..., W⁽ᴸ⁾, b⁽ᴸ⁾}
- Training example (x, y)
- Loss function L(ŷ, y)

**Goal**: Compute ∂L/∂θ for all parameters

**Naive Approach**:

- Compute ∂L/∂W⁽ˡ⁾ᵢⱼ independently for each parameter
- Complexity: O(n²) per parameter → O(n⁴) total (prohibitive!)

**Backpropagation**:

- Use chain rule to share computations
- Complexity: O(n²) total (feasible!)

### Key Insight

**Chain Rule Application**:

```
∂L/∂W⁽ˡ⁾ = (∂L/∂z⁽ˡ⁾) · (∂z⁽ˡ⁾/∂W⁽ˡ⁾)
```

If we compute ∂L/∂z⁽ˡ⁾ once, we can use it for all weights in layer l.

**Recursive Structure**:

```
∂L/∂z⁽ˡ⁾ can be computed from ∂L/∂z⁽ˡ⁺¹⁾

This allows backward propagation through the network
```

## Computational Graph View

### Graph Structure

**Nodes**: Variables and operations

- Input nodes: x
- Intermediate nodes: z⁽ˡ⁾, a⁽ˡ⁾
- Output node: L
- Parameter nodes: W⁽ˡ⁾, b⁽ˡ⁾

**Edges**: Data flow

- Forward: from inputs to output
- Backward: from output to inputs (gradient flow)

### Forward and Backward Passes

**Forward Pass**:

```
x → z⁽¹⁾ → a⁽¹⁾ → z⁽²⁾ → a⁽²⁾ → ... → z⁽ᴸ⁾ → a⁽ᴸ⁾ → L
```

**Backward Pass**:

```
∂L/∂L ← ∂L/∂z⁽ᴸ⁾ ← ∂L/∂a⁽ᴸ⁻¹⁾ ← ∂L/∂z⁽ᴸ⁻¹⁾ ← ... ← ∂L/∂z⁽¹⁾
       (=1)
```

## Mathematical Derivation

### Notation

**Layer l**:

- Pre-activation: z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
- Activation: a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)
- Dimensions:
  - a⁽ˡ⁾ ∈ ℝⁿˡ (n_l neurons in layer l)
  - W⁽ˡ⁾ ∈ ℝⁿˡ ˣ ⁿˡ⁻¹
  - b⁽ˡ⁾ ∈ ℝⁿˡ

**Gradient Notation**:

- δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ (error term for layer l)

### Output Layer (Layer L)

**Loss Gradient**:

```
∂L/∂z⁽ᴸ⁾ = ∂L/∂a⁽ᴸ⁾ ⊙ ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾
         = ∂L/∂a⁽ᴸ⁾ ⊙ f'⁽ᴸ⁾(z⁽ᴸ⁾)
```

where ⊙ denotes element-wise multiplication

**Common Cases**:

**Softmax + Categorical Cross-Entropy**:

```
∂L/∂z⁽ᴸ⁾ = a⁽ᴸ⁾ - y
```

(remarkably simple!)

**Sigmoid + Binary Cross-Entropy**:

```
∂L/∂z⁽ᴸ⁾ = a⁽ᴸ⁾ - y
```

(also simple!)

**Linear + MSE**:

```
∂L/∂z⁽ᴸ⁾ = a⁽ᴸ⁾ - y
```

(pattern emerges!)

### Hidden Layers (Layers l < L)

**Chain Rule Application**:

```
∂L/∂z⁽ˡ⁾ = ∂L/∂a⁽ˡ⁾ ⊙ ∂a⁽ˡ⁾/∂z⁽ˡ⁾
         = ∂L/∂a⁽ˡ⁾ ⊙ f'⁽ˡ⁾(z⁽ˡ⁾)
```

**Computing ∂L/∂a⁽ˡ⁾**:

```
a⁽ˡ⁾ affects L through z⁽ˡ⁺¹⁾:

∂L/∂a⁽ˡ⁾ = ∂L/∂z⁽ˡ⁺¹⁾ · ∂z⁽ˡ⁺¹⁾/∂a⁽ˡ⁾
```

**Since z⁽ˡ⁺¹⁾ = W⁽ˡ⁺¹⁾a⁽ˡ⁾ + b⁽ˡ⁺¹⁾**:

```
∂z⁽ˡ⁺¹⁾/∂a⁽ˡ⁾ = W⁽ˡ⁺¹⁾ᵀ
```

**Therefore**:

```
∂L/∂a⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ · ∂L/∂z⁽ˡ⁺¹⁾
```

**Complete Formula for Hidden Layer**:

```
δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ = ((W⁽ˡ⁺¹⁾)ᵀ · δ⁽ˡ⁺¹⁾) ⊙ f'⁽ˡ⁾(z⁽ˡ⁾)
```

This is the **backpropagation equation** - gradients flow backward!

### Weight Gradients

**Gradient w.r.t. Weights**:

```
∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
```

**Since z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾**:

For element Wᵢⱼ⁽ˡ⁾:

```
∂zᵢ⁽ˡ⁾/∂Wᵢⱼ⁽ˡ⁾ = aⱼ⁽ˡ⁻¹⁾
```

**Matrix Form**:

```
∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
```

**Dimensionality Check**:

```
δ⁽ˡ⁾ ∈ ℝⁿˡ (column vector)
a⁽ˡ⁻¹⁾ ∈ ℝⁿˡ⁻¹ (column vector)
(a⁽ˡ⁻¹⁾)ᵀ ∈ ℝ¹ ˣ ⁿˡ⁻¹ (row vector)

δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ ∈ ℝⁿˡ ˣ ⁿˡ⁻¹ ✓
```

### Bias Gradients

**Gradient w.r.t. Bias**:

```
∂L/∂b⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂b⁽ˡ⁾
```

**Since z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾**:

```
∂z⁽ˡ⁾/∂b⁽ˡ⁾ = I (identity matrix)
```

**Therefore**:

```
∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

## Backpropagation Algorithm

### Single Sample Algorithm

**Input**:

- Training example: (x, y)
- Network parameters: {W⁽ˡ⁾, b⁽ˡ⁾}ₗ₌₁ᴸ
- Activation functions: {f⁽ˡ⁾}ₗ₌₁ᴸ

**Output**:

- Gradients: {∂L/∂W⁽ˡ⁾, ∂L/∂b⁽ˡ⁾}ₗ₌₁ᴸ

**Algorithm**:

```
BACKPROPAGATION(x, y):
    // ========== FORWARD PASS ==========
    a⁽⁰⁾ ← x
    
    FOR l = 1 TO L:
        z⁽ˡ⁾ ← W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
        a⁽ˡ⁾ ← f⁽ˡ⁾(z⁽ˡ⁾)
        CACHE(z⁽ˡ⁾, a⁽ˡ⁾)  // Need for backward pass
    
    // ========== COMPUTE LOSS ==========
    L ← LOSS(a⁽ᴸ⁾, y)
    
    // ========== BACKWARD PASS ==========
    // Output layer gradient
    δ⁽ᴸ⁾ ← ∂L/∂z⁽ᴸ⁾  // Depends on loss and activation
    
    FOR l = L TO 1:
        // Compute gradients for this layer
        ∂L/∂W⁽ˡ⁾ ← δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ
        ∂L/∂b⁽ˡ⁾ ← δ⁽ˡ⁾
        
        // Propagate gradient to previous layer (if not first layer)
        IF l > 1:
            δ⁽ˡ⁻¹⁾ ← ((W⁽ˡ⁾)ᵀ · δ⁽ˡ⁾) ⊙ f'⁽ˡ⁻¹⁾(z⁽ˡ⁻¹⁾)
    
    RETURN {∂L/∂W⁽ˡ⁾, ∂L/∂b⁽ˡ⁾}ₗ₌₁ᴸ
```

### Batch Backpropagation

**For m examples simultaneously**:

**Input**:

- Batch: X ∈ ℝⁿ⁰ ˣ ᵐ (each column is one example)
- Labels: Y ∈ ℝⁿᴸ ˣ ᵐ

**Modifications**:

```
BATCH_BACKPROPAGATION(X, Y):
    // ========== FORWARD PASS ==========
    A⁽⁰⁾ ← X
    
    FOR l = 1 TO L:
        Z⁽ˡ⁾ ← W⁽ˡ⁾ · A⁽ˡ⁻¹⁾ + b⁽ˡ⁾  // Broadcasting bias
        A⁽ˡ⁾ ← f⁽ˡ⁾(Z⁽ˡ⁾)
        CACHE(Z⁽ˡ⁾, A⁽ˡ⁾)
    
    // ========== BACKWARD PASS ==========
    dZ⁽ᴸ⁾ ← A⁽ᴸ⁾ - Y  // Example: softmax + cross-entropy
    
    FOR l = L TO 1:
        // Average gradients across batch
        ∂L/∂W⁽ˡ⁾ ← (1/m) · dZ⁽ˡ⁾ · (A⁽ˡ⁻¹⁾)ᵀ
        ∂L/∂b⁽ˡ⁾ ← (1/m) · SUM_COLUMNS(dZ⁽ˡ⁾)
        
        IF l > 1:
            dZ⁽ˡ⁻¹⁾ ← ((W⁽ˡ⁾)ᵀ · dZ⁽ˡ⁾) ⊙ f'⁽ˡ⁻¹⁾(Z⁽ˡ⁻¹⁾)
    
    RETURN {∂L/∂W⁽ˡ⁾, ∂L/∂b⁽ˡ⁾}ₗ₌₁ᴸ
```

## Activation Function Derivatives

### Common Derivatives (Element-wise)

**ReLU**:

```
f(z) = max(0, z)
f'(z) = { 1  if z > 0
        { 0  if z ≤ 0
```

**Sigmoid**:

```
f(z) = σ(z) = 1/(1 + e⁻ᶻ)
f'(z) = σ(z)(1 - σ(z)) = f(z)(1 - f(z))
```

**tanh**:

```
f(z) = tanh(z)
f'(z) = 1 - tanh²(z) = 1 - f(z)²
```

**Leaky ReLU**:

```
f(z) = { z    if z > 0
       { αz   if z ≤ 0

f'(z) = { 1   if z > 0
        { α   if z ≤ 0
```

**ELU**:

```
f(z) = { z            if z > 0
       { α(eᶻ - 1)    if z ≤ 0

f'(z) = { 1       if z > 0
        { f(z)+α  if z ≤ 0
```

### Softmax Derivative

**Special Case**: Vector function (not element-wise)

For softmax output a = softmax(z):

```
∂aᵢ/∂zⱼ = { aᵢ(1 - aᵢ)   if i = j
          { -aᵢaⱼ        if i ≠ j
```

**With Cross-Entropy Loss**:
The combination simplifies:

```
∂L/∂z = a - y
```

This should be computed directly, not via Jacobian.

## Computational Complexity

### Forward Pass

**Per Layer**:

- Matrix multiplication: O(nˡ · nˡ⁻¹)
- Activation: O(nˡ)
- Total per layer: O(nˡ · nˡ⁻¹)

**Full Network**:

```
O(Σₗ nˡ · nˡ⁻¹)
```

### Backward Pass

**Per Layer**:

- δ⁽ˡ⁾ · (a⁽ˡ⁻¹⁾)ᵀ: O(nˡ · nˡ⁻¹)
- (W⁽ˡ⁾)ᵀ · δ⁽ˡ⁾: O(nˡ · nˡ⁻¹)
- Element-wise operations: O(nˡ)
- Total per layer: O(nˡ · nˡ⁻¹)

**Full Network**:

```
O(Σₗ nˡ · nˡ⁻¹)
```

**Key Insight**: Backward pass has same complexity as forward pass!

### Batch Processing

**With batch size m**:

- Forward: O(m · Σₗ nˡ · nˡ⁻¹)
- Backward: O(m · Σₗ nˡ · nˡ⁻¹)

**Memory**: O(m · Σₗ nˡ) for storing activations

## Gradient Checking

### Numerical Gradient

**Finite Difference Approximation**:

```
∂L/∂θ ≈ [L(θ + ε) - L(θ - ε)] / (2ε)

where ε is small (typically 10⁻⁷)
```

**Two-sided difference** (shown above) is more accurate than one-sided.

### Gradient Check Algorithm

```
GRADIENT_CHECK(θ, ∂L/∂θ):
    ε ← 10⁻⁷
    
    FOR each parameter θᵢ:
        // Compute numerical gradient
        θ_plus ← θ with θᵢ incremented by ε
        θ_minus ← θ with θᵢ decremented by ε
        
        numerical_grad ← [L(θ_plus) - L(θ_minus)] / (2ε)
        
        // Compare with backprop gradient
        difference ← |numerical_grad - ∂L/∂θᵢ|
        relative_error ← difference / max(|numerical_grad|, |∂L/∂θᵢ|)
        
        IF relative_error > 10⁻⁵:
            WARNING: Gradient incorrect for parameter i
    
    overall_difference ← ||numerical_grad - ∂L/∂θ|| / ||numerical_grad||
    
    IF overall_difference < 10⁻⁷:
        EXCELLENT: Gradient implementation correct
    ELSE IF overall_difference < 10⁻⁵:
        OK: Gradient probably correct
    ELSE:
        ERROR: Gradient implementation likely incorrect
```

**Important Notes**:

- Gradient checking is slow (2×parameters forward passes)
- Only use for debugging, not during training
- Turn off regularization when checking
- Use only few examples for checking

## Common Implementation Issues

### Dimensionality Errors

**Common Mistakes**:

```
❌ dW = dZ · A  // Wrong: shapes don't match
✓  dW = dZ · Aᵀ  // Correct

❌ db = dZ  // Wrong if dZ is for batch
✓  db = mean(dZ, axis=1)  // Correct: average over batch
```

**Always Check Shapes**:

```
W⁽ˡ⁾: (nˡ, nˡ⁻¹)
b⁽ˡ⁾: (nˡ, 1) or (nˡ,)
dW⁽ˡ⁾: same shape as W⁽ˡ⁾
db⁽ˡ⁾: same shape as b⁽ˡ⁾
```

### Numerical Instability

**Issues**:

1. **Vanishing gradients**: Gradients become very small in early layers
2. **Exploding gradients**: Gradients become very large
3. **NaN/Inf**: Overflow in computations

**Solutions**:

- Proper weight initialization
- Gradient clipping
- Batch normalization
- Careful activation function choice
- Numerically stable loss implementations

### Cache Management

**Must Cache**:

- All activations a⁽ˡ⁾ (needed for weight gradients)
- All pre-activations z⁽ˡ⁾ (needed for activation derivatives)

**Memory Consideration**:

- For large networks, caching is memory-intensive
- Trade-off: Can recompute activations instead (checkpointing)

## Variations and Extensions

### Momentum

**Concept**: Use previous gradients to inform current update

**Update Rule**:

```
v ← β·v + (1-β)·∂L/∂θ
θ ← θ - α·v

where:
- v: velocity (moving average of gradients)
- β: momentum coefficient (typically 0.9)
```

### Per-Parameter Adaptive Learning Rates

**Adam, RMSprop**: Different learning rates per parameter

- Requires storing additional state
- More complex backprop (parameter-dependent updates)

### Gradient Accumulation

**For very large batches**:

```
1. Forward/backward on mini-batch 1 → accumulate gradients
2. Forward/backward on mini-batch 2 → accumulate gradients
...
n. Update parameters with accumulated gradients
```

## Automatic Differentiation

### Forward Mode vs Reverse Mode

**Forward Mode** (forward accumulation):

- Compute derivatives alongside function evaluation
- Efficient when: few inputs, many outputs
- Complexity: O(n × outputs)

**Reverse Mode** (backpropagation):

- Compute function first, then derivatives backward
- Efficient when: many inputs, few outputs
- Complexity: O(inputs)
- **This is what we use for neural networks**

### Computational Graph

**Automatic Differentiation Systems**:

- Build computational graph during forward pass
- Apply chain rule automatically during backward pass
- Examples: TensorFlow, PyTorch autograd

**Benefits**:

- Correct by construction
- Handle arbitrary computations
- Support dynamic graphs

## Summary

### Key Equations

**Forward** (layer l):

```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)
```

**Backward** (layer l):

```
δ⁽ˡ⁾ = ((W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾) ⊙ f'⁽ˡ⁾(z⁽ˡ⁾)
∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

### Implementation Requirements

1. **Forward Pass**: Compute and cache activations
2. **Loss Computation**: Compute loss and initial gradient
3. **Backward Pass**: Propagate gradients backward
4. **Gradient Storage**: Store gradients for all parameters
5. **Numerical Stability**: Careful implementations
6. **Batch Support**: Vectorized operations
7. **Gradient Checking**: Debugging tool

### Critical Points

- Backprop is just the chain rule, systematically applied
- Same computational complexity as forward pass
- Requires caching intermediate values
- Must handle numerical stability carefully
- Gradient checking is essential for debugging

---

**Previous**: [Loss Functions](./04-loss-functions.md)  
**Next**: [Optimization Algorithms](./06-optimization-algorithms.md)
