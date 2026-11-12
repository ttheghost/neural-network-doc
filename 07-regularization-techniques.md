# Regularization Techniques

## Overview

Regularization techniques prevent overfitting and improve generalization by constraining the model's capacity to memorize training data. This document covers theoretical foundations and specifications for implementing various regularization methods.

## Overfitting and Underfitting

### The Bias-Variance Trade-off

**Bias**: Error from incorrect assumptions in learning algorithm

- High bias → underfitting
- Model too simple to capture patterns

**Variance**: Error from sensitivity to small fluctuations in training data

- High variance → overfitting
- Model too complex, memorizes noise

**Total Error**:

```
Error = Bias² + Variance + Irreducible Error
```

**Goal**: Find sweet spot between bias and variance

### Detecting Overfitting

**Training vs Validation Performance**:

```
Underfitting:    Train loss high, Val loss high
Good fit:        Train loss low,  Val loss low (similar to train)
Overfitting:     Train loss low,  Val loss high (gap increases)
```

**Warning Signs**:

- Validation loss stops decreasing while training loss continues
- Gap between training and validation loss grows
- Perfect or near-perfect training accuracy
- Poor performance on new data

## L1 and L2 Regularization

### L2 Regularization (Weight Decay)

**Mathematical Form**:

```
J_regularized(θ) = J(θ) + (λ/2)·Σ||W⁽ˡ⁾||² 

where:
- J(θ): original loss
- λ: regularization strength (hyperparameter)
- ||W||²: sum of squared weights (Frobenius norm squared)
```

**Note**: Typically, **bias terms are not regularized**

**Gradient**:

```
∇J_regularized = ∇J + λ·W
```

**Update Rule**:

```
W ← W - α·(∇J + λ·W)
  = (1 - αλ)·W - α·∇J
```

**Effect**:

- Penalizes large weights
- Forces weights toward zero
- Smooth weight distribution
- Geometric interpretation: spherical constraint

**Regularization Strength**:

- λ = 0: No regularization
- Small λ (10⁻⁵ - 10⁻⁴): Light regularization
- Medium λ (10⁻³ - 10⁻²): Moderate regularization
- Large λ (> 10⁻²): Strong regularization (may underfit)

**Bayesian Interpretation**:

- Equivalent to Gaussian prior on weights
- MAP estimation with Gaussian prior

### L1 Regularization

**Mathematical Form**:

```
J_regularized(θ) = J(θ) + λ·Σ|W⁽ˡ⁾|
```

**Gradient** (subgradient at 0):

```
∇J_regularized = ∇J + λ·sign(W)

where sign(w) = { +1  if w > 0
                { -1  if w < 0
                {  0  if w = 0
```

**Update Rule**:

```
W ← W - α·(∇J + λ·sign(W))
```

**Effect**:

- Promotes sparsity (many weights → exactly 0)
- Feature selection
- Geometric interpretation: L1 ball constraint (diamond-shaped)

**Sparsity Mechanism**:

- Constant penalty regardless of weight magnitude
- Larger weights don't get preferential treatment
- Pushes small weights to exactly zero

**Use Cases**:

- Feature selection needed
- Interpretability important
- Sparse models desired

### L1 vs L2 Comparison

| Property | L1 | L2 |
|----------|----|----|
| **Sparsity** | Creates sparse solutions | No sparsity |
| **Feature Selection** | Yes (implicit) | No |
| **Derivative** | Constant (for w≠0) | Proportional to weight |
| **Geometry** | Diamond constraint | Spherical constraint |
| **Large Weights** | Not penalized more | Heavily penalized |
| **Common Use** | Feature selection | Default regularization |

### Elastic Net (L1 + L2)

**Mathematical Form**:

```
J = J_original + λ₁·Σ|W| + (λ₂/2)·Σ||W||²
```

**Benefits**:

- Combines L1 sparsity with L2 stability
- Better than L1 when features are correlated
- More stable than L1 alone

**Hyperparameters**:

- λ₁: L1 strength
- λ₂: L2 strength
- Often parameterized as: α (overall strength), ρ (L1 ratio)

## Dropout

### Concept

**Idea**: Randomly "drop" (zero out) neurons during training

**Training**:

- Each neuron kept with probability p (typically 0.5-0.8)
- Different random mask each training iteration
- Forces network to learn redundant representations

**Inference**:

- Use all neurons
- Scale outputs by keep probability p

### Mathematical Formulation

**Training** (layer l):

```
// Generate mask
m⁽ˡ⁾ ← Bernoulli(p)  // 1 with prob p, 0 with prob (1-p)

// Apply mask
a⁽ˡ⁾ ← (a⁽ˡ⁾ ⊙ m⁽ˡ⁾) / p

where:
- ⊙: element-wise multiplication
- division by p: inverted dropout (scaling at train time)
```

**Inference**:

```
// Use full network, no dropout
a⁽ˡ⁾ ← f⁽ˡ⁾(z⁽ˡ⁾)
```

### Inverted Dropout

**Standard Dropout**:

- No scaling during training
- Scale by p during inference
- **Problem**: Inference code must know about dropout

**Inverted Dropout** (preferred):

- Scale by 1/p during training
- No scaling inference
- **Benefit**: Inference code unchanged

**Inverted Implementation**:

```
TRAINING:
    mask ← Bernoulli(p)
    activation ← activation * mask / p

INFERENCE:
    activation ← activation  // unchanged
```

### Dropout Rates

**Hidden Layers**:

- Typical: p = 0.5 (drop 50%)
- Range: 0.3 to 0.8

**Input Layer**:

- Lighter dropout: p = 0.8-0.9 (drop 10-20%)
- Sometimes no dropout

**Output Layer**:

- No dropout (always use all output neurons)

**Convolutional Layers**:

- Often p = 0.9 or no dropout
- Spatial dropout (drop entire feature maps) sometimes used

### Why Dropout Works

**Ensemble Interpretation**:

- Each dropout mask creates a different network
- Training samples many "thinned" networks
- Inference approximates ensemble average
- With n neurons, 2ⁿ possible networks (exponentially large ensemble)

**Co-adaptation Prevention**:

- Neurons can't rely on specific other neurons
- Must learn robust features
- Reduces complex co-adaptations

**Noise Regularization**:

- Adds noise to hidden representations
- Smooth regularizing effect

**Bayesian Interpretation**:

- Approximate Bayesian inference
- Dropout approximates uncertainty

### Implementation Considerations

**Training Mode Flag**:

```
Network must track training vs inference mode
```

**Dropout Placement**:

- Applied after activation, before next layer
- Typically on dense (fully-connected) layers
- Less common in convolutional layers

**Batch Normalization + Dropout**:

- Order matters: typically BatchNorm → Activation → Dropout
- Some debate on optimal order

## Batch Normalization

### Concept

**Problem**: Internal covariate shift

- Distribution of layer inputs changes during training
- Slows down learning
- Requires careful initialization and small learning rates

**Solution**: Normalize layer inputs to stable distribution

### Algorithm

**Training** (for mini-batch B):

For layer l activation z:

```
// Compute batch statistics
μ_B ← (1/m)·Σᵢ∈B z⁽ⁱ⁾
σ²_B ← (1/m)·Σᵢ∈B (z⁽ⁱ⁾ - μ_B)²

// Normalize
ẑ ← (z - μ_B) / √(σ²_B + ε)

// Scale and shift (learnable parameters)
z_out ← γ·ẑ + β

where:
- μ_B, σ²_B: batch mean and variance
- ε: numerical stability (e.g., 10⁻⁵)
- γ, β: learnable scale and shift parameters
```

**Inference**:

```
Use population statistics (running averages):

// Maintained during training:
μ_running ← momentum·μ_running + (1-momentum)·μ_B
σ²_running ← momentum·σ²_running + (1-momentum)·σ²_B

// At inference:
ẑ ← (z - μ_running) / √(σ²_running + ε)
z_out ← γ·ẑ + β
```

### Placement in Network

**Typical Order**:

```
Linear/Conv → BatchNorm → Activation → (Dropout)

OR

Linear/Conv → Activation → BatchNorm → (Dropout)
```

**Most Common**: BatchNorm before activation

- Normalizes input to activation function
- Helps with saturating activations (sigmoid, tanh)

**With ReLU**: BatchNorm after may also work well

### Benefits

1. **Higher Learning Rates**: More stable gradients
2. **Faster Convergence**: Smoother optimization landscape
3. **Regularization**: Acts as mild regularizer (noise from batch statistics)
4. **Less Sensitive to Initialization**: Activations automatically normalized
5. **Reduces need for Dropout**: Built-in regularization effect

### Hyperparameters

**Learnable** (per feature):

- γ: scale
- β: shift

**Fixed**:

- ε: stability constant (10⁻⁵ typical)
- momentum: for running statistics (0.9-0.99 typical)

### Computational Considerations

**Additional Parameters**: 2 × (number of features)

- Usually negligible compared to weights

**Additional Computation**:

- Forward: compute mean, variance, normalize
- Backward: gradient through normalization
- Modest overhead

**Memory**: Store batch statistics, gradients for γ and β

### Variants

**Layer Normalization**:

- Normalize across features (not batch)
- Better for RNNs, small batches
- No distinction between training/inference

**Instance Normalization**:

- Normalize each instance independently
- Used in style transfer

**Group Normalization**:

- Divide features into groups, normalize within groups
- Good for small batches

## Early Stopping

### Concept

**Idea**: Stop training when validation performance stops improving

**Algorithm**:

```
best_val_loss ← ∞
patience_counter ← 0

FOR each epoch:
    val_loss ← VALIDATE()
    
    IF val_loss < best_val_loss:
        best_val_loss ← val_loss
        SAVE_MODEL()  // Save best model
        patience_counter ← 0
    ELSE:
        patience_counter ← patience_counter + 1
    
    IF patience_counter >= patience:
        STOP_TRAINING()
        RESTORE_BEST_MODEL()
```

### Hyperparameters

**Patience**: Number of epochs to wait for improvement

- Small (5-10): Aggressive stopping, risk of premature stop
- Medium (10-20): Balanced
- Large (20-50): Conservative, allows temporary plateaus

**Minimum Delta**: Minimum improvement to count

```
IF val_loss < (best_val_loss - min_delta):
    // Improvement detected
```

Helps ignore noise in validation metrics

### Benefits

- Automatic stopping criterion
- Regularization effect (limits training time)
- Prevents overfitting without manual intervention

### Considerations

- Requires validation set
- May stop before finding better solution
- Patience setting is critical

## Data Augmentation

### Concept

**Idea**: Artificially expand training data with transformed versions

**Effect**:

- Larger effective dataset
- Model sees more variation
- Improves generalization
- Particularly powerful for limited data

### Image Augmentation Techniques

**Geometric Transforms**:

- Random crops
- Horizontal/vertical flips
- Rotations (small angles)
- Scaling
- Shearing
- Translation

**Color Transforms**:

- Brightness adjustment
- Contrast adjustment
- Saturation changes
- Hue shifts

**Noise**:

- Gaussian noise
- Salt and pepper noise

**Advanced**:

- Cutout / Random erasing
- MixUp (blend two images)
- CutMix (paste patches)

### Domain-Specific Augmentation

**Text**:

- Synonym replacement
- Random insertion/deletion
- Back-translation
- Paraphrasing

**Audio**:

- Time stretching
- Pitch shifting
- Adding noise
- Speed perturbation

**Time Series**:

- Jittering
- Scaling
- Window slicing
- Adding noise

### Implementation Considerations

**Apply Online** (during training):

- Generate augmented data on-the-fly
- Different augmentation each epoch
- No storage overhead

**Augmentation Probability**:

- Not all transforms applied every time
- Typically 50% probability per transform

**Validation/Test Sets**:

- **No augmentation** (use clean data for evaluation)
- Or: use test-time augmentation (TTA) for ensemble

## Label Smoothing

### Concept

**Problem**: Hard targets (one-hot) overconfident predictions

**Standard**: y = [0, 0, 1, 0, 0]  (one-hot)

**Label Smoothing**:

```
y_smooth = (1 - ε)·y + ε/K

where:
- ε: smoothing parameter (typically 0.1)
- K: number of classes
```

**Example** (K=5, ε=0.1):

```
Original: [0, 0, 1, 0, 0]
Smoothed: [0.02, 0.02, 0.92, 0.02, 0.02]
```

### Benefits

- Prevents overconfidence
- Improves model calibration
- Better generalization (empirically)
- Works well with large models

### Effect on Loss

**Cross-Entropy with Label Smoothing**:

```
L= -(1-ε)·log(p_correct) - (ε/K)·Σ log(pᵢ)

Encourages:
- High probability for correct class
- Non-zero probability for other classes
```

## Ensemble Methods

### Concept

**Idea**: Combine predictions from multiple models

**Wisdom of Crowds**: Diverse models make different errors

### Ensemble Approaches

**Independent Training**:

```
Train multiple models with:
- Different random initializations
- Different architectures
- Different hyperparameters
- Different data subsets (bagging)

Prediction: Average or vote
```

**Snapshot Ensembles**:

```
Save models at different points during training
Use cyclic learning rate to visit different minima
Ensemble snapshots
```

**Stochastic Weight Averaging (SWA)**:

```
Average weights from different points in training
Single model, ensemble-like benefits
```

### Prediction Aggregation

**Classification**:

- Majority vote (hard voting)
- Average probabilities (soft voting) - often better

**Regression**:

- Mean of predictions
- Median (more robust)

### Trade-offs

**Benefits**:

- Improved accuracy
- Better calibration
- Reduced variance

**Costs**:

- Training time (multiple models)
- Inference time (multiple forward passes)
- Storage (multiple models)

## Other Regularization Techniques

### Noise Injection

**Add Noise to**:

- Inputs (data augmentation)
- Weights (Bayesian interpretation)
- Gradients (improves optimization)
- Hidden layers (similar to dropout)

### Max-Norm Constraints

**Constraint**: Limit weight vector magnitudes

```
After each update:
IF ||W_row|| > c:
    W_row ← W_row · (c / ||W_row||)

Prevents weights from growing too large
```

**Benefits**:

- Controls capacity
- Can use higher learning rates
- Works well with dropout

### Stochastic Depth

**Concept**: Randomly skip layers during training

**Implementation**:

- With probability p, skip layer (use identity)
- Forces network to learn with variable depth
- Regularizes deep networks

## Regularization Selection Guide

### When to Use Each Technique

**L2 Regularization**:

- Default choice for weight regularization
- Small computational overhead
- Works well for most problems

**Dropout**:

- Large fully-connected layers
- When overfitting is severe
- Computational overhead acceptable

**Batch Normalization**:

- Almost always beneficial
- Default in modern architectures
- Especially important for deep networks

**Data Augmentation**:

- Limited training data
- Domain allows meaningful transforms
- Image/audio/text tasks

**Early Stopping**:

- Always recommended
- Free regularization
- Saves training time

**Label Smoothing**:

- Large models
- Overconfident predictions
- Image classification

### Combining Techniques

**Typical Modern Network**:

```
- L2 regularization (weight decay)
- Batch normalization
- Data augmentation
- Early stopping
- Optional: Dropout (if still overfitting)
- Optional: Label smoothing (for classification)
```

**Too Much Regularization**:

- Model underfits
- Training and validation loss both high
- Solution: Reduce regularization strength

## Summary

### Key Takeaways

1. **Multiple complementary techniques**: Use several together
2. **Data augmentation is powerful**: Especially with limited data
3. **Batch Normalization is standard**: Almost always beneficial
4. **Early Stopping is free**: Always monitor validation performance
5. **Start simple, add as needed**: Don't over-regularize initially

### Implementation Checklist

**Essential**:

- L2 regularization (weight decay)
- Dropout
- Batch normalization
- Early stopping

**Recommended**:

- Data augmentation (domain-specific)
- Label smoothing
- Learning rate scheduling (also acts as regularization)

**Advanced**:

- Ensembles
- Stochastic Weight Averaging
- Mixup/CutMix (image data)

---

**Previous**: [Optimization Algorithms](./06-optimization-algorithms.md)  
**Next**: [Network Architectures](./08-network-architectures.md)
