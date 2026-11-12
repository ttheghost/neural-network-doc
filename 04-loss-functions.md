# Loss Functions

## Overview

Loss functions (also called cost functions or objective functions) quantify how well a neural network's predictions match the true labels. They provide the optimization target for training and drive the learning process through their gradients.

## Fundamental Concepts

### Loss vs Cost

**Loss**: Error for a single training example

```
L(ŷ, y) = measure of discrepancy between prediction ŷ and true label y
```

**Cost**: Average loss over entire dataset

```
J = (1/m) Σᵢ L(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
```

**Terminology Note**: Often used interchangeably, but technically:

- Loss: per-example
- Cost: over dataset

### Desirable Properties

1. **Differentiability**: Must be differentiable w.r.t. predictions for gradient-based optimization
2. **Convexity**: Convex functions guarantee global minimum (rare in deep learning)
3. **Appropriate Scale**: Loss magnitude should be reasonable
4. **Robustness**: Should not be overly sensitive to outliers (problem-dependent)
5. **Interpretability**: Easier to debug and tune

### Role in Learning

**Training Loop**:

```
1. Forward pass: compute predictions ŷ
2. Compute loss: L(ŷ, y)
3. Backward pass: compute ∂L/∂θ (gradients)
4. Update parameters: θ ← θ - α·∂L/∂θ
```

The gradient ∂L/∂θ tells us how to adjust parameters to reduce loss.

## Classification Loss Functions

### Binary Cross-Entropy (Log Loss)

**Use Case**: Binary classification (two classes)

**Mathematical Form**:

```
L(ŷ, y) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

where:
- y ∈ {0, 1} (true label)
- ŷ ∈ (0, 1) (predicted probability)
```

**For m examples**:

```
J = -(1/m) Σᵢ [y⁽ⁱ⁾·log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾)·log(1-ŷ⁽ⁱ⁾)]
```

**Gradient w.r.t. prediction**:

```
∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
      = (ŷ - y) / (ŷ(1-ŷ))
```

**With sigmoid output**:
If ŷ = σ(z), the gradient simplifies beautifully:

```
∂L/∂z = ŷ - y
```

**Range**: [0, ∞)

- Perfect prediction (ŷ = y): L = 0
- Wrong prediction: L → ∞

**Properties**:

- Penalizes confident wrong predictions heavily
- Convex in z (pre-activation) when using sigmoid
- Derived from maximum likelihood estimation

**Numerical Stability**:

```
Problem: log(0) = -∞

Solution 1: Clip predictions
ŷ_stable = clip(ŷ, ε, 1-ε)  where ε ≈ 1e-7

Solution 2: Compute from logits directly (preferred)
```

**Logit-based Stable Implementation**:

```
Given z (pre-activation logit):
L = max(z, 0) - z·y + log(1 + exp(-|z|))

This is numerically stable and avoids computing sigmoid explicitly
```

### Categorical Cross-Entropy

**Use Case**: Multi-class classification (mutually exclusive classes)

**Mathematical Form**:

```
L(ŷ, y) = -Σₖ yₖ·log(ŷₖ)

where:
- y is one-hot encoded: yₖ ∈ {0,1}, Σₖ yₖ = 1
- ŷ is predicted probability: ŷₖ ∈ (0,1), Σₖ ŷₖ = 1
- k is the class index
```

**If y is one-hot** (only one class is true):

```
L = -log(ŷⱼ)  where j is the true class index

In other words, loss = negative log probability of correct class
```

**For m examples**:

```
J = -(1/m) Σᵢ Σₖ y⁽ⁱ⁾ₖ·log(ŷ⁽ⁱ⁾ₖ)
```

**Gradient w.r.t. prediction**:

```
∂L/∂ŷₖ = -yₖ/ŷₖ
```

**With softmax output**:
If ŷ = softmax(z), gradient w.r.t. logits:

```
∂L/∂zₖ = ŷₖ - yₖ

For one-hot y:
∂L/∂zⱼ = ŷⱼ - 1  (for true class j)
∂L/∂zₖ = ŷₖ      (for other classes k ≠ j)
```

**Properties**:

- Encourages correct class probability → 1
- Unbounded loss for wrong predictions
- Convex in logits with softmax
- Maximum likelihood for categorical distribution

**Numerical Stability**:

```
Softmax + cross-entropy should be fused:

Naive:
  ŷ = softmax(z)
  L = -log(ŷⱼ)

Stable (combined):
  L = -zⱼ + log(Σₖ exp(zₖ))
  = log(Σₖ exp(zₖ)) - zⱼ
  = log(Σₖ exp(zₖ - c)) + c - zⱼ  (where c = max(z))
```

### Sparse Categorical Cross-Entropy

**Use Case**: Multi-class classification when labels are integers (not one-hot)

**Mathematical Form**:

```
Same as categorical cross-entropy, but:
- y is integer class label: y ∈ {0, 1, ..., K-1}
- ŷ is still probability distribution

L = -log(ŷy)  (y indexes into ŷ)
```

**Advantage**:

- Memory efficient (no need to one-hot encode)
- Faster computation
- Identical gradient

**Implementation Note**:
Functionally identical to categorical cross-entropy, just different input format.

### Binary Cross-Entropy with Logits

**Mathematical Form**:

```
Given logit z (pre-sigmoid activation):
L = max(z, 0) - z·y + log(1 + exp(-|z|))
```

**Advantages**:

- Numerically stable
- Avoids computing sigmoid explicitly
- More accurate gradients

**This is equivalent to**:

```
ŷ = sigmoid(z)
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

But numerically superior
```

### Hinge Loss (SVM Loss)

**Use Case**: Binary classification, maximum margin

**Mathematical Form**:

```
L(ŷ, y) = max(0, 1 - y·ŷ)

where:
- y ∈ {-1, +1} (note: different encoding than cross-entropy)
- ŷ ∈ ℝ (raw score, not probability)
```

**Properties**:

- Loss is 0 when y·ŷ ≥ 1 (correct with margin)
- Linear penalty for violations
- Not differentiable at y·ŷ = 1

**Gradient**:

```
∂L/∂ŷ = { -y  if y·ŷ < 1
        { 0   otherwise
```

**Squared Hinge Loss**:

```
L = max(0, 1 - y·ŷ)²

Smooth, differentiable everywhere
```

**Use Cases**:

- Support Vector Machines
- When maximum margin is desired
- Less common in deep learning than cross-entropy

### Multi-Class Hinge Loss

**Mathematical Form**:

```
L = Σₖ≠ⱼ max(0, ŷₖ - ŷⱼ + Δ)

where:
- j is the true class
- k are other classes
- Δ is a margin (typically 1.0)
```

**Interpretation**:

- Penalizes when score of wrong class is within margin of correct class
- Encourages correct class score to exceed others by at least Δ

## Regression Loss Functions

### Mean Squared Error (MSE) / L2 Loss

**Use Case**: Regression, predicting continuous values

**Mathematical Form**:

```
L(ŷ, y) = (ŷ - y)²

For m examples:
J = (1/m) Σᵢ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
```

**Root Mean Squared Error (RMSE)**:

```
RMSE = √[(1/m) Σᵢ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²]

Same minimizer as MSE, but in same units as target
```

**Gradient**:

```
∂L/∂ŷ = 2(ŷ - y)

(Factor of 2 often absorbed into learning rate)
```

**Properties**:

- Heavily penalizes large errors (quadratic)
- Sensitive to outliers
- Differentiable everywhere
- Assumes Gaussian error distribution

**Advantages**:

- Simple, interpretable
- Smooth gradient
- Well-understood properties

**Disadvantages**:

- Outlier sensitive (large errors dominate)
- Quick gradient growth (can cause instability)

### Mean Absolute Error (MAE) / L1 Loss

**Use Case**: Regression, robust to outliers

**Mathematical Form**:

```
L(ŷ, y) = |ŷ - y|

For m examples:
J = (1/m) Σᵢ |ŷ⁽ⁱ⁾ - y⁽ⁱ⁾|
```

**Gradient**:

```
∂L/∂ŷ = { +1  if ŷ > y
        { -1  if ŷ < y
        (undefined at ŷ = y, use 0 in practice)
```

**Properties**:

- Linear penalty for errors
- Robust to outliers
- Not differentiable at zero error
- Assumes Laplacian error distribution

**Advantages**:

- Robust to outliers
- Constant gradient magnitude

**Disadvantages**:

- Non-smooth at perfect prediction
- Can be slow to converge near minimum
- Constant gradient (no adaptive scaling)

### Huber Loss

**Use Case**: Regression, compromise between MSE and MAE

**Mathematical Form**:

```
L(ŷ, y) = { ½(ŷ - y)²              if |ŷ - y| ≤ δ
          { δ|ŷ - y| - ½δ²         otherwise

where δ is a threshold parameter
```

**Gradient**:

```
∂L/∂ŷ = { (ŷ - y)              if |ŷ - y| ≤ δ
        { δ·sign(ŷ - y)        otherwise
```

**Properties**:

- Quadratic for small errors (smooth, like MSE)
- Linear for large errors (robust, like MAE)
- Differentiable everywhere

**Advantages**:

- Best of both worlds: smooth + robust
- Less sensitive to outliers than MSE
- Smoother than MAE

**Disadvantages**:

- Extra hyperparameter δ

**Selection of δ**:

- δ ≈ 1.0 is common default
- Can be tuned based on error distribution

### Log-Cosh Loss

**Mathematical Form**:

```
L(ŷ, y) = log(cosh(ŷ - y))
        = log[(eˣ + e⁻ˣ)/2]  where x = ŷ - y
```

**Gradient**:

```
∂L/∂ŷ = tanh(ŷ - y)
```

**Properties**:

- Approximately quadratic for small errors
- Approximately linear for large errors (like Huber)
- Twice differentiable (smoother than Huber)

**Advantages**:

- Very smooth (C² continuous)
- Robust to outliers
- No hyperparameter (unlike Huber)

**Disadvantages**:

- More expensive to compute  (hyperbolic functions)

### Quantile Loss

**Use Case**: Quantile regression, predicting specific percentiles

**Mathematical Form**:

```
L(ŷ, y) = { τ(y - ŷ)      if y ≥ ŷ
          { (τ-1)(y - ŷ)  if y < ŷ

where τ ∈ (0, 1) is the desired quantile
```

**Special Cases**:

- τ = 0.5: Median regression (equivalent to MAE)
- τ = 0.9: 90th percentile
- τ = 0.1: 10th percentile

**Properties**:

- Asymmetric penalty
- Predicts conditional quantiles
- Robust to outliers

**Use Cases**:

- When certain types of errors are more costly
- Uncertainty estimation
- Risk-sensitive predictions

## Multi-Label Loss Functions

### Binary Cross-Entropy (Multi-Label)

**Use Case**: Multi-label classification (independent labels)

**Mathematical Form**:

```
Each label treated independently with sigmoid:

L = -(1/K) Σₖ [yₖ·log(ŷₖ) + (1-yₖ)·log(1-ŷₖ)]

where:
- yₖ ∈ {0,1} for each label
- ŷₖ ∈ (0,1) for each label
- Labels are independent (multiple can be 1)
```

**Comparison**:

- Multi-class (softmax): Exactly one class is true (Σyₖ = 1)
- Multi-label (sigmoid): Any subset of labels can be true

## Advanced Loss Functions

### Focal Loss

**Use Case**: Classification with class imbalance

**Mathematical Form**:

```
FL(ŷ, y) = -α(1 - ŷ)ᵞ·log(ŷ)  for y = 1
         = -α·ŷᵞ·log(1 - ŷ)     for y = 0

where:
- α: class weighting factor
- γ: focusing parameter (typically 2)
```

**For multi-class**:

```
FL = -α(1 - ŷⱼ)ᵞ·log(ŷⱼ)  where j is true class
```

**Properties**:

- Down-weights easy examples (high confidence correct predictions)
- Focuses training on hard examples
- Addresses class imbalance

**Intuition**:

```
When ŷⱼ is high (easy example): (1-ŷⱼ)ᵞ ≈ 0, loss ≈ 0
When ŷⱼ is low (hard example): (1-ŷⱼ)ᵞ ≈ 1, full loss
```

**Use Cases**:

- Object detection
- Highly imbalanced datasets
- When rare classes are important

### Triplet Loss

**Use Case**: Metric learning, embeddings

**Mathematical Form**:

```
L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)

where:
- f(a): embedding of anchor
- f(p): embedding of positive (same class as anchor)
- f(n): embedding of negative (different class)
- margin: minimum desired separation
```

**Goal**:

- Anchor should be closer to positive than to negative
- By at least the specified margin

**Properties**:

- Learns relative distances
- Does not require explicit class labels during testing
- Produces embedding space

**Use Cases**:

- Face recognition
- Similarity learning
- Few-shot learning

### Contrastive Loss

**Use Case**: Siamese networks, similarity learning

**Mathematical Form**:

```
L = (1-Y)·½D² + Y·½max(0, margin - D)²

where:
- D = ||f(x₁) - f(x₂)||: distance between embeddings
- Y ∈ {0,1}: 0 if similar, 1 if dissimilar
- margin: minimum distance for dissimilar pairs
```

**Goal**:

- Similar pairs: minimize distance
- Dissimilar pairs: enforce distance ≥ margin

## Loss Function Selection Guide

### Classification

**Binary (2 classes)**:

- **Standard**: Binary Cross-Entropy
- **Imbalanced**: Focal Loss or weighted Binary Cross-Entropy
- **Maximum margin**: Hinge Loss

**Multi-class (mutually exclusive)**:

- **Standard**: Categorical Cross-Entropy
- **Imbalanced**: Focal Loss or class weights
- **Integer labels**: Sparse Categorical Cross-Entropy

**Multi-label (independent)**:

- **Standard**: Binary Cross-Entropy per label

### Regression

**Standard conditions**:

- **Default**: MSE (Mean Squared Error)
- **Outliers present**: MAE or Huber
- **Smooth + robust**: Huber or Log-Cosh
- **Specific percentile**: Quantile Loss

### Special Tasks

**Embeddings/Similarity**:

- Triplet Loss
- Contrastive Loss

**Generative Models**:

- Reconstruction: MSE or Binary Cross-Entropy
- GANs: Various adversarial losses

## Regularization in Loss

### L2 Regularization (Weight Decay)

**Mathematical Form**:

```
J_regularized = J + λ·Σ||W⁽ˡ⁾||²

where:
- J: original loss
- λ: regularization strength
- ||W⁽ˡ⁾||²: sum of squared weights
```

**Effect**:

- Penalizes large weights
- Encourages simpler models
- Reduces overfitting

### L1 Regularization

**Mathematical Form**:

```
J_regularized = J + λ·Σ|W⁽ˡ⁾|
```

**Effect**:

- Encourages sparsity (many weights → 0)
- Feature selection
- Creates sparse networks

### Elastic Net (L1 + L2)

**Mathematical Form**:

```
J = J_original + λ₁·Σ|W| + λ₂·Σ||W||²
```

**Effect**:

- Combines benefits of L1 and L2
- Sparse + small weights

## Implementation Considerations

### Numerical Stability

**Critical Points**:

1. **Logarithms**: Never directly log(ŷ), always use log-tricks
2. **Exponentials**: Softmax needs max-subtraction
3. **Division**: Check for division by zero
4. **Fused Operations**: Combine activation + loss when possible

**Example - Sigmoid + BCE**:

```
Don't:
  σ = sigmoid(z)
  loss = -y·log(σ) - (1-y)·log(1-σ)

Do:
  loss = max(z,0) - z·y + log(1 + exp(-abs(z)))
```

### Gradient Computation

**Efficient Patterns**:

**Softmax + Cross-Entropy**:

```
Combined gradient: ∂L/∂z = ŷ - y

Individual gradients:
  ∂L/∂ŷ = -y/ŷ
  ∂ŷ/∂z = Jacobian of softmax
  Combined: much simpler!
```

**Sigmoid + BCE**:

```
Combined gradient: ∂L/∂z = ŷ - y
```

### Vectorization

**Batch Loss Computation**:

```
For m examples:
- Compute loss for all examples simultaneously
- Average: J = (1/m)·sum(losses)
- Gradient: average gradients across batch
```

**Matrix Form Benefits**:

- GPU acceleration
- Cache efficiency
- Simpler code

## Summary Table

| Loss | Task | Properties | Robustness |
|------|------|-----------|------------|
| Binary Cross-Entropy | Binary Classification | Probabilistic, unbounded | Low (outliers hurt) |
| Categorical Cross-Entropy | Multi-class | Probabilistic, encourages confidence | Low |
| Focal Loss | Imbalanced Classification | Focuses on hard examples | Medium |
| MSE | Regression | Smooth, penalizes large errors | Low (outlier sensitive) |
| MAE | Regression | Robust, constant gradient | High (outlier robust) |
| Huber | Regression | Smooth + robust | High |
| Hinge | Classification | Maximum margin | Medium |

## Implementation Checklist

**Essential Loss Functions**:

- Binary Cross-Entropy (with logits)
- Categorical Cross-Entropy (with logits, sparse version)
- MSE
- MAE

**Recommended**:

- Huber Loss
- Focal Loss (for imbalanced data)

**For Each Loss**:

1. Forward computation: L(ŷ, y)
2. Gradient: ∂L/∂ŷ or ∂L/∂z (if fused with activation)
3. Numerically stable implementation
4. Batch-wise computation
5. Reduction options (mean, sum, none)

---

**Previous**: [Activation Functions](./03-activation-functions.md)  
**Next**: [Backpropagation Algorithm](./05-backpropagation.md)
