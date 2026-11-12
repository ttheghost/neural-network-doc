# Activation Functions

## Overview

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without activation functions, multiple layers would collapse into a single linear transformation, eliminating the benefits of depth.

## Purpose and Properties

### Why Non-linearity Matters

**Without Activation Functions**:

```
Layer 1: z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾
Layer 2: z⁽²⁾ = W⁽²⁾z⁽¹⁾ + b⁽²⁾
       = W⁽²⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾
       = (W⁽²⁾W⁽¹⁾)x + (W⁽²⁾b⁽¹⁾ + b⁽²⁾)
       = W̃x + b̃  (equivalent to single layer!)
```

**With Activation Functions**:

```
Layer 1: a⁽¹⁾ = f(W⁽¹⁾x + b⁽¹⁾)
Layer 2: a⁽²⁾ = f(W⁽²⁾a⁽¹⁾ + b⁽²⁾)
       (Cannot be reduced to single linear operation)
```

### Desirable Properties

1. **Non-linearity**: Must introduce non-linear transformations
2. **Differentiability**: Should be differentiable (or subdifferentiable) for gradient-based optimization
3. **Computational Efficiency**: Fast to compute (applied billions of times)
4. **Gradient Flow**: Should not cause vanishing or exploding gradients
5. **Range**: Output range should be appropriate for the problem
6. **Zero-Centered**: Often beneficial for optimization
7. **Monotonicity**: Usually (but not always) desirable

### Gradient Flow Concerns

**Vanishing Gradients**:

- Gradient becomes very small
- Deep layers don't learn effectively
- Common with sigmoid/tanh in deep networks

**Exploding Gradients**:

- Gradient becomes very large
- Unstable training
- Can cause NaN values

## Common Activation Functions

### Sigmoid (Logistic)

**Mathematical Form**:

```
σ(x) = 1 / (1 + e⁻ˣ)
```

**Range**: (0, 1)

**Derivative**:

```
σ'(x) = σ(x)(1 - σ(x))
```

**Properties**:

- Smooth S-shaped curve
- Outputs interpretable as probabilities
- Squashes input to (0, 1)

**Advantages**:

- Clear probabilistic interpretation
- Smooth gradient
- Bounded output

**Disadvantages**:

- Vanishing gradient problem for |x| large
- Not zero-centered (outputs always positive)
- Expensive exponential computation

**Use Cases**:

- Output layer for binary classification
- Gate mechanisms (LSTM, GRU)
- Historically used in hidden layers (now rare)

**Numerical Stability**:

```
For large negative x, e⁻ˣ overflows
Stable implementation:
  if x ≥ 0:
    return 1 / (1 + exp(-x))
  else:
    return exp(x) / (1 + exp(x))
```

### Hyperbolic Tangent (tanh)

**Mathematical Form**:

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ) = 2σ(2x) - 1
```

**Range**: (-1, 1)

**Derivative**:

```
tanh'(x) = 1 - tanh²(x)
```

**Properties**:

- S-shaped, similar to sigmoid
- Zero-centered (outputs can be negative)
- Symmetric around origin

**Advantages**:

- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid (derivative range: 0 to 1)
- Smooth

**Disadvantages**:

- Still suffers from vanishing gradients
- Expensive computation

**Use Cases**:

- Hidden layers (better than sigmoid, but ReLU preferred now)
- Recurrent networks (LSTM, GRU)
- Output layer when outputs should be in (-1, 1)

### ReLU (Rectified Linear Unit)

**Mathematical Form**:

```
ReLU(x) = max(0, x) = { x  if x > 0
                      { 0  if x ≤ 0
```

**Range**: [0, ∞)

**Derivative**:

```
ReLU'(x) = { 1  if x > 0
           { 0  if x ≤ 0
           (undefined at x = 0, conventionally set to 0 or 1)
```

**Properties**:

- Piece-wise linear
- Not bounded above
- Not differentiable at x=0 (but subdifferentiable)

**Advantages**:

- **Very fast to compute**: Simple max operation
- **No vanishing gradient** for positive values
- **Sparse activation**: About 50% of neurons are zero
- Empirically converges faster than sigmoid/tanh
- Simple gradient: 0 or 1

**Disadvantages**:

- **Dying ReLU problem**: Neurons can permanently output 0
- Not zero-centered
- Unbounded activation (can grow very large)

**Dying ReLU**:

```
If a neuron gets large negative weighted sum:
- Gradient is 0
- No weight updates
- Neuron "dies" and never activates again
```

**Use Cases**:

- **Default choice** for hidden layers in feedforward networks
- Convolutional neural networks
- Most modern architectures

### Leaky ReLU

**Mathematical Form**:

```
LeakyReLU(x) = { x      if x > 0
               { αx     if x ≤ 0
```

where α is a small constant (typically 0.01)

**Range**: (-∞, ∞)

**Derivative**:

```
LeakyReLU'(x) = { 1   if x > 0
                { α   if x ≤ 0
```

**Properties**:

- Allows small gradient when x < 0
- Prevents dying ReLU problem

**Advantages**:

- Fixes dying ReLU problem
- Still computationally efficient
- Non-zero gradient everywhere

**Disadvantages**:

- Extra hyperparameter α
- Results not always better than ReLU

**Variants**:

- **Parametric ReLU (PReLU)**: α is learned during training
- **Randomized Rleaky ReLU (RReLU)**: α is random during training

### ELU (Exponential Linear Unit)

**Mathematical Form**:

```
ELU(x) = { x                if x > 0
         { α(eˣ - 1)        if x ≤ 0
```

where α > 0 (typically α = 1)

**Range**: (-α, ∞)

**Derivative**:

```
ELU'(x) = { 1              if x > 0
          { α·eˣ           if x ≤ 0
          { (or: ELU(x) + α if x ≤ 0)
```

**Properties**:

- Smooth everywhere
- Negative saturation
- Mean activation closer to zero

**Advantages**:

- No dying ReLU problem
- Negative values push mean activations toward zero
- Smooth gradient transition
- Better learning characteristics empirically

**Disadvantages**:

- Exponential computation (slower than ReLU)
- Extra hyperparameter α

**Use Cases**:

- When training instability observed with ReLU
- When better performance needed and speed is not critical

### GELU (Gaussian Error Linear Unit)

**Mathematical Form**:

```
GELU(x) = x · Φ(x)
```

where Φ(x) is the cumulative distribution function of standard normal distribution

**Approximation**:

```
GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])

or

GELU(x) ≈ x · σ(1.702x)
```

**Properties**:

- Smooth, non-monotonic
- Stochastic regularizer interpretation
- Weights inputs by their value

**Advantages**:

- Used in state-of-the-art architectures (BERT, GPT)
- Combines properties of dropout and ReLU
- Smooth gradients

**Disadvantages**:

- More computationally expensive
- More complex to implement

**Use Cases**:

- Transformer models
- Modern NLP architectures
- State-of-the-art computer vision models

### Swish / SiLU (Sigmoid Linear Unit)

**Mathematical Form**:

```
Swish(x) = x · σ(βx)
```

where β is a learnable parameter (often β = 1)

When β = 1, also called SiLU:

```
SiLU(x) = x · σ(x) = x / (1 + e⁻ˣ)
```

**Derivative**:

```
Swish'(x) = Swish(x) + σ(x)(1 - Swish(x))
```

**Properties**:

- Smooth, non-monotonic
- Self-gated
- Unbounded above, bounded below

**Advantages**:

- Empirically outperforms ReLU in deep networks
- Smooth (helps optimization)
- Non-monotonicity can be beneficial

**Disadvantages**:

- More expensive than ReLU (sigmoid computation)
- Less interpretable

**Use Cases**:

- Deep neural networks where performance matters
- Found via neural architecture search

### Softmax

**Mathematical Form**:

```
For input vector z = [z₁, z₂, ..., zₖ]:

softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
```

**Range**: (0, 1) for each element, sum = 1

**Properties**:

- Vector-to-vector function (not element-wise)
- Outputs sum to 1 (probability distribution)
- Monotonic: larger inputs → larger outputs

**Derivative** (Jacobian):

```
∂softmax(z)ᵢ/∂zⱼ = { softmax(z)ᵢ(1 - softmax(z)ᵢ)  if i = j
                   { -softmax(z)ᵢ·softmax(z)ⱼ       if i ≠ j
```

**Numerical Stability**:

```
Problem: e^(zᵢ) can overflow for large zᵢ

Solution: Subtract max before exp:
c = max(z)
softmax(z)ᵢ = e^(zᵢ - c) / Σⱼ e^(zⱼ - c)

This is mathematically equivalent but stable:
e^(zᵢ - c) / Σⱼ e^(zⱼ - c) = [e^(-c)·e^(zᵢ)] / [e^(-c)·Σⱼ e^(zⱼ)]
                            = e^(zᵢ) / Σⱼ e^(zⱼ)
```

**Use Cases**:

- **Output layer for multi-class classification** (most common)
- Attention mechanisms
- Any time probability distribution needed

**Softmax vs Sigmoid**:

- Softmax: Multi-class (mutually exclusive classes)
- Sigmoid: Binary or multi-label (independent classes)

### Softplus

**Mathematical Form**:

```
softplus(x) = ln(1 + eˣ)
```

**Range**: (0, ∞)

**Derivative**:

```
softplus'(x) = eˣ / (1 + eˣ) = σ(x)
```

**Properties**:

- Smooth approximation of ReLU
- Always positive
- Strictly positive gradients

**Relationship to ReLU**:

```
For large x: softplus(x) ≈ x (like ReLU)
For x near 0: smoother transition
```

**Use Cases**:

- Smooth alternative to ReLU
- When strictly positive outputs needed
- Variance parameterization in probabilistic models

## Activation Function Selection Guide

### Hidden Layers

**Default Choice**: ReLU

- Fast, effective, widely used
- Good starting point for most problems

**When to Use Alternatives**:

- **LeakyReLU/PReLU**: If dying ReLU is observed
- **ELU**: For better performance, willing to trade speed
- **GELU/Swish**: For state-of-the-art models, especially transformers
- **tanh**: For RNNs, LSTMs (traditional choice)

### Output Layers

**Binary Classification**: Sigmoid

- Outputs in (0, 1)
- Interpret as probability

**Multi-class Classification**: Softmax

- Outputs sum to 1
- Probability distribution over classes

**Multi-label Classification**: Sigmoid (on each output)

- Independent probabilities
- Multiple classes can be true simultaneously

**Regression**:

- **No activation** (linear): Unrestricted range
- **ReLU**: Non-negative outputs only
- **Sigmoid**: Bounded to (0, 1)
- **tanh**: Bounded to (-1, 1)
- **Softplus**: Strictly positive outputs

## Gradient Behavior

### Gradient Magnitude Comparison

**For x = 1**:

```
sigmoid'(1)   ≈ 0.196
tanh'(1)      ≈ 0.420
ReLU'(1)      = 1.000
LeakyReLU'(1) = 1.000
ELU'(1)       = 1.000
```

**For x = -1**:

```
sigmoid'(-1)    ≈ 0.196
tanh'(-1)       ≈ 0.420
ReLU'(-1)       = 0
LeakyReLU'(-1)  = 0.01 (for α=0.01)
ELU'(-1)        ≈ 0.368 (for α=1)
```

### Saturation

**Definition**: Region where gradient becomes very small

**Saturating Functions**:

- Sigmoid: saturates for |x| > 4
- tanh: saturates for |x| > 2

**Non-saturating Functions**:

- ReLU: no upper saturation
- LeakyReLU: no saturation in either direction

## Implementation Considerations

### Computational Cost

**Ranked by Speed** (fastest to slowest):

1. ReLU, LeakyReLU: Simple comparison and multiplication
2. tanh, sigmoid: May use lookup tables or approximations
3. ELU: Exponential computation
4. GELU, Swish: Most complex
5. Softmax: Sum across all outputs

### Memory Requirements

**Forward Pass**:

- Store input x (needed for backward pass)
- Some activations can store output instead (e.g., sigmoid, tanh)

**Backward Pass**:

```
For activation a = f(z):
Gradient: ∂L/∂z = (∂L/∂a) ⊙ f'(z)

Many functions have f'(z) = g(f(z)):
- sigmoid: f'(z) = f(z)(1 - f(z))
- tanh: f'(z) = 1 - f(z)²
- Can compute gradient using only output (save memory)
```

### Numerical Stability

**Sigmoid Stability**:

```
Standard: 1/(1 + exp(-x))
Problem: exp(-x) overflows for x << 0

Stable version:
if x >= 0:
    return 1 / (1 + exp(-x))
else:
    ez = exp(x)
    return ez / (1 + ez)
```

**Softmax Stability**:

```
Always subtract max:
softmax(x - max(x))
```

**ELU Stability**:

```
For large negative x, exp(x) underflows to 0
Result: α(0 - 1) = -α  (stable)
```

## Advanced Topics

### Learnable Activations

**Parametric ReLU**:

```
PReLU(x) = { x     if x > 0
           { αx    if x ≤ 0

where α is learned per-channel
```

**Maxout**:

```
maxout(x)ᵢ = max(wᵢ₁ᵀx + bᵢ₁, wᵢ₂ᵀx + bᵢ₂, ...)

Learns activation function
Can approximate any convex function
```

### Activation Function Design

**Automatic Discovery**:

- Neural Architecture Search (NAS)
- Can discover novel activation functions
- Examples: Swish was discovered this way

**Design Principles**:

1. Preserve gradient flow
2. Computational efficiency
3. Smooth (for optimization)
4. Empirical performance

## Summary Table

| Function | Range | Gradient Flow | Speed | Use Case |
|----------|-------|---------------|-------|----------|
| Sigmoid | (0,1) | Vanishing | Medium | Binary output |
| tanh | (-1,1) | Vanishing | Medium | RNNs, symmetric |
| ReLU | [0,∞) | Good/Dying | **Fast** | **Default hidden** |
| LeakyReLU | (-∞,∞) | Good | **Fast** | Fix dying ReLU |
| ELU | (-α,∞) | Good | Slow | Better performance |
| GELU | (-∞,∞) | Good | Slow | Transformers |
| Swish | (-∞,∞) | Good | Medium | Deep networks |
| Softmax | (0,1) sum=1 | Good | Medium | **Multi-class output** |

## Implementation Checklist

For a neural network library, implement:

**Essential**:

- ReLU
- Sigmoid
- Softmax
- tanh (if supporting RNNs)

**Recommended**:

- LeakyReLU / PReLU
- ELU
- Linear (identity, for output layers)

**Advanced**:

- GELU
- Swish/SiLU
- Softplus

**For each activation**:

- Forward evaluation: f(x)
- Backward gradient: f'(x) or method to compute ∂L/∂x given ∂L/∂f(x)
- Numerical stability
- Efficient implementation (vectorized, in-place when possible)

---

**Previous**: [Neural Network Fundamentals](./02-neural-network-fundamentals.md)  
**Next**: [Loss Functions](./04-loss-functions.md)
