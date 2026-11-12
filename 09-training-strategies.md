# Training Strategies

## Overview

This document covers practical strategies for training neural networks effectively, from data preparation to debugging and monitoring. These are essential techniques that complement the theoretical foundations.

## Data Preparation

### Data Splitting

**Standard Splits**:

```
Training Set: 60-80%  - For learning parameters
Validation Set: 10-20% - For hyperparameter tuning, early stopping
Test Set: 10-20% - For final unbiased evaluation
```

**Important Principles**:

- Test set NEVER used for any training decisions
- Validation set for all tuning (not test set!)
- Splits should be representative of data distribution

**Special Cases**:

**Small Datasets** (< 10,000 examples):

- Larger validation/test splits (e.g., 70/15/15)
- Consider cross-validation

**Large Datasets** (> 1M examples):

- Smaller validation/test percentages acceptable
- 98/1/1 split can be acceptable

**Time Series**:

- Respect temporal order
- Train on past, validate/test on future
- No random shuffling!

### Cross-Validation

**K-Fold Cross-Validation**:

```
FOR k = 1 TO K:
    Use fold k as validation
    Use remaining K-1 folds as training
    Train and evaluate
    
Average performance across all folds
```

**Benefits**:

- Better use of limited data
- More robust evaluation
- Reduces variance in performance estimates

**Disadvantages**:

- K times more training
- Expensive for neural networks

**When to Use**:

- Small datasets
- Critical applications requiring robust estimates
- Hyperparameter search

### Data Normalization

**Purpose**: Standardize input features to similar scales

**Methods**:

**Min-Max Scaling**:

```
x_scaled = (x - x_min) / (x_max - x_min)

Result: [0, 1] range
```

**Standardization (Z-score)**:

```
x_standardized = (x - μ) / σ

where:
- μ: mean
- σ: standard deviation

Result: ~N(0, 1) distribution (mean 0, std 1)
```

**Important**:

- Compute statistics on training set ONLY
- Apply same transformation to validation/test
- Store scaling parameters for inference

**Per-Channel Normalization** (images):

```
For each color channel:
x_c = (x_c - μ_c) / σ_c

Preserves relative color information
```

### Class Imbalance

**Problem**: Unequal class representation

**Example**:

```
Fraud detection: 99% non-fraud, 1% fraud
Model predicting always "non-fraud" achieves 99% accuracy!
```

**Solutions**:

**Class Weights**:

```
weight_class_i = n_total / (n_classes × n_class_i)

Or simply: Higher weight for minority class

Loss: weighted_loss = weight[y] × loss
```

**Oversampling Minority Class**:

- Duplicate or augment minority examples
- Balance dataset artificially

**Undersampling Majority Class**:

- Randomly remove majority examples
- Risk: losing information

**SMOTE** (Synthetic Minority Over-sampling):

- Generate synthetic examples
- Interpolate between minority class neighbors

**Focal Loss**:

- Automatically down-weights easy examples
- Focuses on hard examples

### Data Augmentation Pipeline

**Training Time**:

```
FOR each batch:
    Load examples
    Apply random augmentations
    Normalize
    Create batch tensor
```

**Validation/Test Time**:

```
Load examples
Normalize (no augmentation!)
Create batch tensor
```

## Weight Initialization

### Why Initialization Matters

**Poor Initialization Consequences**:

- Vanishing gradients: All gradients → 0
- Exploding gradients: All gradients → ∞
- Symmetry: All neurons learn same features
- Slow convergence or failure to learn

### Initialization Strategies

**Zero Initialization**:

```
W = 0

❌ BAD: All neurons identical, symmetry problem
```

**Random Small Values**:

```
W ~ Uniform(-0.01, 0.01)

❌ Issue: May cause vanishing activations in deep networks
```

**Xavier/Glorot Initialization**:

```
W ~ Uniform(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

Or Normal Distribution:
W ~ N(0, 2/(n_in + n_out))

Good for: tanh, sigmoid activations
```

**He Initialization**:

```
W ~ N(0, 2/n_in)

Good for: ReLU and its variants (recommended!)
```

**Rationale**:

- Maintain variance of activations across layers
- Prevent vanishing/exploding during forward pass
- Ensure gradients flow during backward pass

**Bias Initialization**:

```
b = 0  (typically)

Or small positive values for ReLU (e.g., 0.01)
```

### Layer-Specific Initialization

**Convolutional Layers**:

- He initialization for filters
- Consider fan-in/fan-out based on kernel size

**Recurrent Networks**:

- Identity or orthogonal initialization for recurrent weights
- Xavier/He for input weights

**Batch Normalization**:

- γ (scale): 1
- β (shift): 0

## Learning Rate Selection

### Finding Good Learning Rate

**Learning Rate Range Test**:

```
ALGORITHM:
1. Start with very small LR (e.g., 1e-7)
2. Train for several iterations
3. Multiply LR by constant each iteration (e.g., 1.1)
4. Plot loss vs LR
5. Select LR where loss decreases fastest
6. Use LR slightly smaller than where loss diverges
```

**Typical Plot**:

```
Loss
  |     \
  |      \___
  |          \___/   (minimum)
  |              \
  |               ︎\︎︎___↑ (divergence)
  |_________________
       Learning Rate (log scale)
       
Choose LR slightly left of minimum
```

### Warmup

**Problem**: Large learning rates at start can destabilize training

**Solution**: Gradually increase LR at beginning

**Linear Warmup**:

```
FOR iteration t = 1 TO warmup_steps:
    LR(t) = LR_max × (t / warmup_steps)

Then use regular schedule
```

**Typical Warmup**: 1000-10000 steps

**Particularly Important For**:

- Large batch training
- Adam optimizer
- Transformer models

## Batch Processing

### Batch Size Selection

**Small Batches** (16-64):

- Noisy gradients
- Regularization effect
- Better generalization
- Slower per epoch
- Lower GPU utilization

**Large Batches** (256-1024+):

- Stable gradients
- Faster per epoch
- Better hardware utilization
- May need LR adjustment
- Risk of sharp minima (worse generalization)

**Practical Considerations**:

- Memory constraints (largest batch that fits)
- Hardware efficiency (saturate GPU)
- Often: 32, 64, 128, 256

### Gradient Accumulation

**For Effective Large Batches**:

**Problem**: Desired batch size doesn't fit in memory

**Solution**: Accumulate gradients over multiple mini-batches

```
ALGORITHM:
Set accumulation_steps = effective_batch_size / mini_batch_size

Initialize: gradients = 0

FOR step = 1 TO accumulation_steps:
    batch = get_mini_batch()
    loss = forward(batch) / accumulation_steps
    gradients += backward(loss)

Apply accumulated gradients
Zero gradients
```

**Effect**: Same as single large batch, but fits in memory

## Monitoring and Debugging

### Metrics to Track

**Essential**:

- Training loss
- Validation loss
- Training accuracy (if classification)
- Validation accuracy

**Important**:

- Learning rate (if using schedule)
- Gradient norms
- Weight norms
- Training throughput (examples/second)

**Advanced**:

- Activation statistics (mean, std per layer)
- Weight update ratios (|Δw| / |w|)
- Attention weights (if applicable)

### Diagnostic Patterns

**Training Loss Patterns**:

**Not Decreasing**:

```
Possible causes:
- Learning rate too low
- Poor initialization
- Bug in implementation
- Model too small
```

**Decreasing Too Slowly**:

```
Possible causes:
- Learning rate too conservative
- Insufficient capacity
- Optimization algorithm not suited
```

**Fluctuating Wildly**:

```
Possible causes:
- Learning rate too high
- Batch size too small
- Numerical instability
```

**Training/Validation Gap**:

**Small Gap** (Good):

```
Training loss ≈ Validation loss
Model generalizes well
```

**Growing Gap** (Overfitting):

```
Training loss << Validation loss
Increase regularization
Get more data
```

**Both High** (Underfitting):

```
Training loss high, Validation loss high
Increase capacity
Train longer
Reduce regularization
```

### Gradient Monitoring

**Gradient Norms**:

```
Monitor ||∇L|| for each layer

Healthy: Gradients in range [0.0001, 10]
Vanishing: Gradients < 0.0001
Exploding: Gradients > 100
```

**Action for Vanishing**:

- Check initialization
- Use ReLU instead of sigmoid/tanh
- Use batch normalization
- Add skip connections

**Action for Exploding**:

- Gradient clipping
- Reduce learning rate
- Check for numerical errors

### Overfitting Diagnosis

**Symptoms**:

- Perfect or near-perfect training accuracy
- Poor validation/test accuracy
- Large gap between train and val loss

**Solutions** (in order of priority):

1. Get more data (best solution)
2. Data augmentation
3. Add regularization (L2, dropout)
4. Reduce model capacity
5. Early stopping
6. Ensemble methods

## Training Best Practices

### Baseline First

**Process**:

```
1. Start simple (small network, standard architecture)
2. Verify setup on tiny dataset (should overfit!)
3. Add complexity gradually
4. Compare to known baselines
```

**Sanity Checks**:

- Can model overfit single batch? (Should!)
- Random initialization: better than random prediction?
- Training loss decreasing?

### Incremental Complexity

**Don't Start With**:

- Huge model
- Complex architecture
- All regularization techniques
- Fancy optimization

**Do Start With**:

- Simple baseline model
- Minimal regularization
- Standard optimizer (Adam)
- Default hyperparameters

**Then Add**:

- More capacity (if underfitting)
- Regularization (if overfitting)
- Data augmentation
- Learning rate schedule

### Reproducibility

**Set Random Seeds**:

```
Set seeds for:
- Random number generator
- NumPy
- Neural network library
- GPU operations (if deterministic mode available)
```

**Track Hyperparameters**:

- All settings in config file or version control
- Random seed
- Data version
- Code version

**Log Everything**:

- Hyperparameters
- Metrics at each epoch
- Model checkpoints
- Final results

## Advanced Training Techniques

### Mixed Precision Training

**Concept**: Use lower precision (FP16) for speed, higher precision (FP32) for stability

**Algorithm**:

```
Store:
- Weights in FP32 (master copy)
- Activations in FP16

Forward pass: FP16
Backward pass: FP16 → gradients in FP16
Convert gradients to FP32
Update FP32 master weights
Copy to FP16 for next iteration
```

**Loss Scaling**:

```
Scale loss by large factor before backward pass
Scale gradients back after backward pass

Prevents gradient underflow in FP16
```

**Benefits**:

- 2x faster training
- 2x less memory
- Larger batches possible

**Requirements**:

- Modern GPU (Volta or newer)
- Careful implementation

### Curriculum Learning

**Concept**: Train on easy examples first, gradually increase difficulty

**Example**:

```
Epoch 1-10: Short sequences
Epoch 11-20: Medium sequences
Epoch 21+: All sequences
```

**Benefits**:

- Faster convergence (empirically)
- Better final performance
- Particularly useful for RNNs

### Transfer Learning

**Concept**: Use pre-trained model as starting point

**Approaches**:

**Feature Extraction**:

```
1. Load pre-trained model
2. Freeze all layers except last
3. Train only final layer on new task
```

**Fine-Tuning**:

```
1. Load pre-trained model
2. Replace final layer for new task
3. Train all layers with small learning rate
```

**Progressive Unfreezing**:

```
1. Train final layer
2. Unfreeze and train last few layers
3. Gradually unfreeze more layers
4. Use smaller LR for earlier layers
```

**Benefits**:

- Much less data needed
- Faster training
- Often better performance
- Particularly powerful for images (ImageNet) and text (BERT, GPT)

## Distributed Training

### Data Parallelism

**Concept**: Split batch across multiple GPUs

**Algorithm**:

```
FOR each GPU:
    Forward pass on mini-batch subset
    Backward pass → local gradients
    
Synchronize gradients across GPUs (all-reduce)
Apply averaged gradients
```

**Benefits**:

- Larger effective batch size
- Faster training (proportional to GPUs)

**Considerations**:

- May need to adjust learning rate
- Communication overhead

### Model Parallelism

**Concept**: Split model across GPUs

**When Needed**:

- Model doesn't fit on single GPU
- Very large models

**Complexity**: More difficult to implement efficiently

## Hyperparameter Tuning

### Hyperparameters to Tune

**Priority Order**:

1. **Learning rate** (most critical!)
2. **Batch size**
3. **Network architecture** (depth, width)
4. **Optimizer** (Adam vs SGD+momentum)
5. **Regularization strength** (L2, dropout rate)
6. **Learning rate schedule**

**Lower Priority**:

- Optimizer parameters (β₁, β₂ for Adam)
- Batch normalization momentum
- Minor architectural choices

### Tuning Strategies

**Grid Search**:

```
Define discrete values for each hyperparameter
Try all combinations

Example:
LR: [0.001, 0.01, 0.1]
Batch: [32, 64, 128]
→ 3 × 3 = 9 experiments
```

**Pros**: Exhaustive
**Cons**: Exponential in number of hyperparameters

**Random Search**:

```
Define distributions for each hyperparameter
Sample randomly

Often: Log-uniform for LR, uniform for others
```

**Pros**: More efficient than grid search
**Cons**: May miss optimal configuration

**Bayesian Optimization**:

```
Use probabilistic model to guide search
Sample where expected improvement is high
```

**Pros**: Most sample-efficient
**Cons**: More complex to implement

**Recommendation**:

- Start with random search
- Use logarithmic scale for LR
- Iteratively narrow ranges

## Troubleshooting Guide

### Loss is NaN

**Causes**:

- Learning rate too high → exploding gradients
- Numerical overflow in loss computation
- Division by zero

**Solutions**:

- Reduce learning rate
- Gradient clipping
- Check loss implementation (numerical stability)

### Loss Not Decreasing

**Causes**:

- Learning rate too low
- Bug in backpropagation
- Model insufficient for task
- Batch size too small

**Solutions**:

- Increase learning rate
- Verify gradients (gradient checking)
- Increase model capacity
- Increase batch size

### Overfitting

**Solutions** (in order):

1. More data / data augmentation
2. Stronger regularization (Dropout, L2)
3. Reduce model size
4. Early stopping
5. Ensemble methods

### Slow Training

**Solutions**:

- Increase batch size (if memory allows)
- Use GPU if available
- Optimize data loading (preprocessing, caching)
- Mixed precision training
- Profile code for bottlenecks

## Summary

### Essential Practices

1. **Proper data splitting**: Train/val/test
2. **Normalize inputs**: Standardize or min-max scaling
3. **Good initialization**: He for ReLU, Xavier for tanh/sigmoid
4. **Start simple**: Baseline first, add complexity
5. **Monitor carefully**: Watch train/val metrics  
6. **Use validation set**: For all tuning decisions
7. **Reproducibility**: Set random seeds, log everything

### Common Pitfalls to Avoid

1. Using test set for any decisions (validation set exists for this!)
2. Not normalizing inputs
3. Poor initialization
4. Learning rate too high or too low
5. Not monitoring validation performance
6. Over-regularizing from the start
7. Not verifying model can overfit

### Key Takeaways

- **Data preparation is critical**
- **Learning rate matters most**
- **Start simple, add complexity**
- **Monitor and debug systematically**
- **Validation set guides all decisions**

---

**Previous**: [Network Architectures](./08-network-architectures.md)  
**Next**: [API Design Principles](./10-api-design-principles.md)
