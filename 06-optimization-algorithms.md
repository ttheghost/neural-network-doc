# Optimization Algorithms

## Overview

Optimization algorithms are responsible for updating network parameters to minimize the loss function. This document covers the theory and specifications of gradient-based optimization methods used in training neural networks.

## The Optimization Problem

### Objective

**Goal**: Find parameters θ* that minimize the loss function

```
θ* = argmin J(θ)
     θ

where J(θ) = (1/m) Σᵢ L(f(x⁽ⁱ⁾; θ), y⁽ⁱ⁾)
```

### Challenges

1. **Non-convexity**: Multiple local minima
2. **High dimensionality**: Millions to billions of parameters
3. **Noise**: Stochastic gradients from mini-batches
4. **Saddle points**: More common than local minima in high dimensions
5. **Ill-conditioning**: Different parameters at different scales

### Gradient-Based Optimization

**Core Principle**: Move in direction of negative gradient

```
θₜ₊₁ = θₜ - α·∇J(θₜ)

where:
- α: learning rate (step size)
- ∇J(θₜ): gradient of loss w.r.t. parameters
```

## Learning Rate

### Role and Selection

**Learning Rate (α)**: Controls update step size

**Too Large**:

- Training unstable
- Overshoots minimum
- Can diverge

**Too Small**:

- Very slow convergence
- Gets stuck in plateaus
- Wastes computational resources

**Typical Range**: 0.001 to 0.1 (problem-dependent)

### Learning Rate Schedules

#### Fixed Learning Rate

```
α(t) = α₀

Simple, but often suboptimal
```

#### Step Decay

```
α(t) = α₀ · γ^⌊t/s⌋

where:
- γ: decay factor (e.g., 0.5)
- s: step interval (e.g., every 10 epochs)
```

#### Exponential Decay

```
α(t) = α₀ · e^(-kt)

or

α(t) = α₀ · γ^t

Smooth continuous decay
```

#### Inverse Time Decay

```
α(t) = α₀ / (1 + kt)

where k controls decay rate
```

#### Polynomial Decay

```
α(t) = α₀ · (1 - t/T)^p

where:
- T: total training steps
- p: polynomial power (typically 1.0)
```

#### Cosine Annealing

```
α(t) = α_min + ½(α_max - α_min)(1 + cos(πt/T))

Smooth oscillation to minimum
```

#### Warm Restarts

```
Periodically reset learning rate to high value
Helps escape local minima
Used in SGDR (Stochastic Gradient Descent with Warm Restarts)
```

## Basic Optimization Algorithms

### Batch Gradient Descent

**Update Rule**:

```
θ ← θ - α·∇J(θ)

where ∇J(θ) computed over entire dataset
```

**Characteristics**:

- Exact gradient direction
- Deterministic updates
- Slow for large datasets
- Guaranteed convergence for convex functions

**Complexity per Update**: O(m × n)

- m: number of examples
- n: number of parameters

**Use Cases**:

- Small datasets
- When exact gradients needed
- Theoretical analysis

### Stochastic Gradient Descent (SGD)

**Update Rule**:

```
FOR each example (xⁱ, yⁱ):
    θ ← θ - α·∇L(f(xⁱ; θ), yⁱ)
```

**Characteristics**:

- High variance updates
- Noisy, but fast
-Can escape shallow local minima (due to noise)
- Requires careful learning rate tuning

**Advantages**:

- Fast updates (one example at a time)
- Online learning possible
- Memory efficient

**Disadvantages**:

- High variance
- Slow convergence
- Difficult to parallelize

### Mini-Batch Gradient Descent

**Update Rule**:

```
FOR each mini-batch B:
    θ ← θ - α·(1/|B|)·Σᵢ∈B ∇L(f(xⁱ; θ), yⁱ)
```

**Characteristics**:

- Balance between batch and stochastic
- Reduced variance vs pure SGD
- Leverages vectorization/parallelization
- **Most common in practice**

**Batch Size Selection**:

**Small Batches** (8-32):

- More noise, more regularization effect
- Better generalization (empirically)
- Less memory

**Large Batches** (128-512 or more):

- More stable gradients
- Better hardware utilization
- Faster training per epoch
- May generalize worse

**Typical Range**: 32, 64, 128, 256

## Momentum-Based Methods

### Momentum

**Motivation**: Accelerate learning in relevant directions, dampen oscillations

**Update Rule**:

```
v ← β·v + (1-β)·∇J(θ)
θ ← θ - α·v

where:
- v: velocity (exponential moving average of gradients)
- β: momentum coefficient (typically 0.9)
```

**Alternative Formulation** (more common):

```
v ← β·v + ∇J(θ)
θ ← θ - α·v
```

**Intuition**:

- Velocity accumulates past gradients
- Accelerates in consistent directions
- Dampens oscillations

**Effect of β**:

- β = 0: No momentum (standard SGD)
- β = 0.9: Standard momentum
- β = 0.99: Heavy momentum

**Advantages**:

- Faster convergence
- Smoother trajectory
- Overcomes some local minima

**Memory**: O(n) additional storage for velocity

### Nesterov Accelerated Gradient (NAG)

**Motivation**: "Look ahead" before computing gradient

**Update Rule**:

```
θ_lookahead ← θ - β·v
v ← β·v + ∇J(θ_lookahead)
θ ← θ - α·v
```

**Intuition**:

- Evaluate gradient at predicted future position
- More responsive to changes
- Can achieve better convergence

**Advantage over standard momentum**:

- Better theoretical convergence rate
- Often faster in practice
- More "intelligent" updates

## Adaptive Learning Rate Methods

### AdaGrad

**Motivation**: Adapt learning rate per parameter based on historical gradients

**Update Rule**:

```
Initialize: G ← 0

At each iteration:
g ← ∇J(θ)  // current gradient
G ← G + g ⊙ g  // accumulate squared gradients
θ ← θ - α·g / (√G + ε)

where:
- ε: small constant for numerical stability (e.g., 10⁻⁸)
- ⊙: element-wise multiplication
- /: element-wise division
```

**Per-Parameter Form**:

```
θᵢ ← θᵢ - α·gᵢ / √(Σₜ gᵢ,ₜ² + ε)

Learning rate decreases with accumulated squared gradients
```

**Characteristics**:

- Smaller updates for frequently updated parameters
- Larger updates for infrequent parameters
- Well-suited for sparse data

**Disadvantages**:

- Accumulator G grows monotonically
- Learning rate continually decreases
- Can stop learning too early

**Memory**: O(n) for accumulator G

### RMSprop

**Motivation**: Fix AdaGrad's aggressive learning rate decay

**Update Rule**:

```
Initialize: v ← 0

At each iteration:
g ← ∇J(θ)
v ← β·v + (1-β)·g ⊙ g  // exponential moving average
θ ← θ - α·g / (√v + ε)

where β typically 0.9
```

**Difference from AdaGrad**:

- Uses exponential moving average instead of sum
- Doesn't accumulate indefinitely
- Learning rate doesn't vanish

**Characteristics**:

- Works well with non-stationary objectives
- Handles sparse gradients
- Often used for RNNs

**Hyperparameters**:

- α: learning rate (0.001 typical)
- β: decay rate (0.9 typical)

**Memory**: O(n) for moving average

### Adam (Adaptive Moment Estimation)

**Motivation**: Combine momentum and adaptive learning rates

**Update Rule**:

```
Initialize: m ← 0, v ← 0, t ← 0

At each iteration:
t ← t + 1
g ← ∇J(θ)

// Update biased first moment estimate
m ← β₁·m + (1-β₁)·g

// Update biased second moment estimate  
v ← β₂·v + (1-β₂)·g ⊙ g

// Bias correction
m̂ ← m / (1 - β₁ᵗ)
v̂ ← v / (1 - β₂ᵗ)

// Update parameters
θ ← θ - α·m̂ / (√v̂ + ε)
```

**Components**:

- m: First moment (mean) - like momentum
- v: Second moment (variance) - like RMSprop
- Bias correction: Adjusts for initialization bias

**Default Hyperparameters**:

- α: 0.001
- β₁: 0.9 (first moment decay)
- β₂: 0.999 (second moment decay)
- ε: 10⁻⁸

**Characteristics**:

- **Most popular optimizer** in practice
- Works well with minimal tuning
- Robust to hyperparameter choices
- Efficient, low memory

**Advantages**:

- Combines benefits of momentum and RMSprop
- Bias correction improves early training
- Works well on wide variety of problems

**Memory**: O(2n) for m and v

### AdamW

**Modification**: Proper weight decay implementation

**Update Rule**:

```
// Adam update as before, but add weight decay:
θ ← θ - α·(m̂ / (√v̂ + ε) + λ·θ)

where:
- λ: weight decay coefficient
```

**Key Difference from Adam with L2 regularization**:

- Decoupled weight decay
- Often better generalization
- Recommended over standard Adam

### Nadam (Nesterov Adam)

**Combination**: Adam + Nesterov momentum

**Update Rule**:

```
Similar to Adam, but uses Nesterov-style momentum in update:
θ ← θ - α·(β₁·m̂ + (1-β₁)·ĝ) / (√v̂ + ε)

where ĝ is bias-corrected gradient
```

**Advantage**: Slightly faster convergence than Adam

## Advanced Techniques

### Gradient Clipping

**Motivation**: Prevent exploding gradients

**Methods**:

**Clip by Value**:

```
g ← clip(g, -threshold, +threshold)

Each gradient component clipped independently
```

**Clip by Norm**:

```
IF ||g|| > threshold:
    g ← g · (threshold / ||g||)

Preserves gradient direction, scales magnitude
```

**Recommended**: Clip by norm (preserves direction)

**Typical Values**: threshold = 1.0 to 5.0

### Weight Decay

**Motivation**: Regularization, prevent overfitting

**L2 Regularization**:

```
J_regularized = J + (λ/2)·Σ θᵢ²

Gradient becomes:
∇J_regularized = ∇J + λ·θ
```

**Weight Decay Implementation**:

```
θ ← θ - α·(∇J + λ·θ)
  = (1 - α·λ)·θ - α·∇J

Direct decay of weights
```

**Typical Values**: λ = 10⁻⁵ to 10⁻³

### Batch Normalization and Optimization

**Effect**: Makes optimization landscape smoother

**Benefits**:

- Allows higher learning rates
- Reduces sensitivity to initialization
- Acts as regularization

**Interaction**: Batch norm changes effective learning rate

## Second-Order Methods

### Newton's Method

**Update Rule**:

```
θ ← θ - H⁻¹·∇J(θ)

where H is Hessian matrix (second derivatives)
```

**Characteristics**:

- Uses curvature information
- Can achieve superlinear convergence
- Quadratic convergence near optimum

**Disadvantages**:

- Hessian computation: O(n²) memory, O(n³) inversion
- Prohibitive for deep learning
- Rarely used directly

### L-BFGS (Limited-memory BFGS)

**Motivation**: Approximate Newton's method with limited memory

**Characteristics**:

- Approximates inverse Hessian
- O(k×n) memory, where k ≈ 10-20
- Deterministic (uses full batch)

**Use Cases**:

- Small to medium networks
- When exact gradients available
- Style transfer, adversarial examples

**Not Common** for large-scale deep learning

## Comparison and Selection Guide

### Optimizer Comparison Table

| Optimizer | Memory | Convergence Speed | Tuning Difficulty | Best Use Case |
|-----------|--------|-------------------|-------------------|---------------|
| SGD | Low | Slow | High | With good tuning, small batches |
| SGD + Momentum | Medium | Medium-Fast | Medium | General purpose, proven |
| Adam | Medium | Fast | Low | **Default choice, most problems** |
| AdamW | Medium | Fast | Low | Adam + better regularization |
| RMSprop | Medium | Fast | Medium | RNNs, non-stationary |
| AdaGrad | Medium | Slow (eventually) | Low | Sparse features |

### Selection Guidelines

**Default Choice**: Adam or AdamW

- Works well out of the box
- Minimal tuning required
- Fast convergence

**When to Use SGD + Momentum**:

- Want best final performance (with lots of tuning)
- Have expertise to tune learning rate schedule
- Using small batch sizes

**When to Use RMSprop**:

- Training RNNs or LSTMs
- Non-stationary problems
- When Adam doesn't work well

**When to Use Specialized**:

- AdaGrad: Sparse data (NLP, recommender systems)
- L-BFGS: Small networks, style transfer
- Second-order: Research, specialized applications

## Practical Considerations

### Hyperparameter Tuning

**Critical Hyperparameters**:

1. Learning rate (most important!)
2. Batch size
3. Momentum coefficient (if using)
4. Weight decay

**Tuning Strategy**:

1. Start with defaults (especially for Adam)
2. Tune learning rate first (logarithmic search)
3. Adjust batch size (hardware-dependent)
4. Fine-tune other parameters if needed

### Learning Rate Finding

**Method**: Learning rate range test

**Algorithm**:

```
1. Start with very small learning rate (e.g., 10⁻⁷)
2. Increase exponentially each batch
3. Plot loss vs learning rate
4. Select learning rate where loss decreases fastest
5. Use 10x smaller than where loss explodes
```

### Monitoring Training

**Key Metrics**:

- Training loss (should decrease smoothly)
- Validation loss (should decrease, watch for overfitting)
- Gradient norms (should be stable)
- Parameter update magnitudes

**Warning Signs**:

- Loss not decreasing: learning rate too low or initialization issue
- Loss exploding: learning rate too high
- Training/validation diverge: overfitting
- Gradients vanishing/exploding: architecture or depth issue

### Batch Size Effects

**Large Batches**:

- More stable gradients
- Better hardware utilization
- Need to scale learning rate (linear scaling rule)
- May generalize worse (sharp minima)

**Small Batches**:

- More noise (regularization effect)
- Better generalization (flat minima)
- More iterations per epoch
- Less efficient

**Linear Scaling Rule**:

```
If batch size increased by k, increase learning rate by k

Example: Batch 32 with α=0.001 → Batch 256 with α=0.008
```

## Implementation Specification

### Optimizer Interface

**Required Operations**:

1. Initialize optimizer state
2. Compute update given gradients
3. Apply update to parameters
4. Update learning rate (if using schedule)

**State Management**:

- Store per-parameter adaptive state (Adam: m, v)
- Track global state (iteration counter, learning rate)
- Support state save/load (checkpointing)

### Numerical Stability

**Critical Points**:

- Division by zero: always add ε to denominators
- Overflow in squared gradients: monitor and clip if needed
- Bias correction: important for Adam variants

**Epsilon Values**:

- Adam: ε = 10⁻⁸ typical
- RMSprop: ε = 10⁻⁸ to 10⁻⁶

## Summary

### Key Takeaways

1. **Adam is the default**: Works well for most problems
2. **Learning rate matters most**: Spend time tuning this
3. **SGD + Momentum can achieve best results**: But requires expertise
4. **Batch size affects convergence**: Tune jointly with learning rate
5. **Monitoring is essential**: Watch losses and gradients

### Essential for Implementation

**Must Have**:

- SGD
- SGD with Momentum
- Adam / AdamW

**Recommended**:

- RMSprop
- Learning rate schedules (step, exponential, cosine)
- Gradient clipping

**Advanced**:

- Nadam
- L-BFGS (if supporting)
- Custom schedulers

---

**Previous**: [Backpropagation Algorithm](./05-backpropagation.md)  
**Next**: [Regularization Techniques](./07-regularization-techniques.md)
