# Neural Network Fundamentals

## Overview

This document introduces the fundamental concepts of neural networks, from biological inspiration to the mathematical models that form the basis of modern deep learning.

## Biological Inspiration

### The Biological Neuron

**Components**:

1. **Dendrites**: Receive signals from other neurons
2. **Cell Body (Soma)**: Processes incoming signals
3. **Axon**: Transmits output signal
4. **Synapses**: Connections between neurons with varying strengths

**Process**:

```
Input signals → Weighted summation → Activation threshold → Output signal
```

**Key Insight**: The strength of connections (synaptic weights) determines how signals propagate through the network.

### Abstraction to Artificial Neurons

**Simplifications**:

- Discrete time steps (not continuous firing)
- Simplified activation model
- Weights represent connection strength
- Bias represents intrinsic excitability

**Difference from Biology**:

- Real neurons are far more complex
- Timing and spike patterns matter in biology
- Artificial neurons are mathematical approximations

## The Perceptron

### Mathematical Model

**Definition**: The simplest artificial neuron model

**Mathematical Form**:

```
Given inputs: x = [x₁, x₂, ..., xₙ]
Weights: w = [w₁, w₂, ..., wₙ]
Bias: b

Weighted sum: z = w·x + b = Σ(wᵢxᵢ) + b

Output: y = f(z)
```

where f is an activation function

### Geometric Interpretation

**Decision Boundary**:

- Perceptron creates a linear decision boundary
- In 2D: a line
- In 3D: a plane
- In nD: a hyperplane

**Equation**:

```
w₁x₁ + w₂x₂ + ... + wₙxₙ + b = 0
```

**Classification**:

- Points on one side: class 0
- Points on other side: class 1

### Limitations

**XOR Problem**:

- Cannot learn XOR function
- XOR is not linearly separable
- Demonstrates need for multiple layers

**Linear Separability Requirement**:

- Perceptron can only solve linearly separable problems
- Many real-world problems are not linearly separable

## Multi-Layer Networks

### Architecture

**Layer Types**:

1. **Input Layer**
   - Receives raw input features
   - No computation, just distribution
   - Size: number of input features

2. **Hidden Layers**
   - Perform intermediate computations
   - Learn hierarchical representations
   - Can have multiple hidden layers (deep networks)

3. **Output Layer**
   - Produces final predictions
   - Size depends on task:
     - Binary classification: 1 or 2 neurons
     - Multi-class: k neurons (k classes)
     - Regression: typically 1 neuron (can be more)

### Mathematical Formulation

**Layer-wise Computation**:

For layer l with:

- Input: **a**⁽ˡ⁻¹⁾ (activation from previous layer)
- Weights: **W**⁽ˡ⁾
- Bias: **b**⁽ˡ⁾
- Activation function: f⁽ˡ⁾

Compute:

```
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾    (pre-activation)
a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)           (activation)
```

**Full Network**:

```
Input: x = a⁽⁰⁾

Layer 1: z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾,  a⁽¹⁾ = f⁽¹⁾(z⁽¹⁾)
Layer 2: z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾, a⁽²⁾ = f⁽²⁾(z⁽²⁾)
...
Layer L: z⁽ᴸ⁾ = W⁽ᴸ⁾a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾, a⁽ᴸ⁾ = f⁽ᴸ⁾(z⁽ᴸ⁾)

Output: ŷ = a⁽ᴸ⁾
```

### Dimensionality

**Layer l Parameters**:

- Input dimension: n⁽ˡ⁻¹⁾
- Output dimension (neurons): n⁽ˡ⁾
- Weight matrix: **W**⁽ˡ⁾ ∈ ℝⁿ⁽ˡ⁾ ˣ ⁿ⁽ˡ⁻¹⁾
- Bias vector: **b**⁽ˡ⁾ ∈ ℝⁿ⁽ˡ⁾

**Total Parameters**:

```
For layer l: n⁽ˡ⁾ × n⁽ˡ⁻¹⁾ + n⁽ˡ⁾ = n⁽ˡ⁾(n⁽ˡ⁻¹⁾ + 1)
```

### Feedforward Process

**Definition**: Forward propagation of data through the network

**Steps**:

1. Present input to input layer
2. Compute each layer sequentially from input to output
3. Each layer's output becomes the next layer's input
4. Final layer produces prediction

**Batch Processing**:
When processing m examples simultaneously:

```
X ∈ ℝⁿ⁽⁰⁾ ˣ ᵐ  (each column is one example)

Z⁽ˡ⁾ = W⁽ˡ⁾A⁽ˡ⁻¹⁾ + b⁽ˡ⁾  (broadcasting bias)
A⁽ˡ⁾ = f⁽ˡ⁾(Z⁽ˡ⁾)
```

**Computational Complexity**:

- Per layer: O(n⁽ˡ⁾ · n⁽ˡ⁻¹⁾ · m) for matrix multiplication
- Highly parallelizable

## Representation Learning

### Feature Hierarchy

**Concept**: Deep networks learn hierarchical representations

**Example in Image Recognition**:

```
Layer 1: Edges and basic shapes
Layer 2: Textures and simple patterns
Layer 3: Object parts (eyes, wheels, etc.)
Layer 4: Complete objects (faces, cars, etc.)
```

**Advantage**:

- Manual feature engineering not required
- Network learns optimal features for the task
- Features are task-specific

### Compositionality

**Principle**: Complex patterns built from simpler ones

**Mathematical View**:

```
y = fₗ(fₗ₋₁(...f₂(f₁(x))))
```

Each function f adds a level of abstraction

**Power of Depth**:

- With d layers and n neurons per layer
- Can represent O(nᵈ) regions in input space
- Shallow network would need O(nᵈ) neurons in single layer

## Universal Approximation Theorem

### Statement

**Theorem**: A feedforward network with:

- Single hidden layer
- Finite number of neurons
- Non-polynomial activation function

can approximate any continuous function on a compact subset of ℝⁿ to arbitrary accuracy.

**Mathematical Form**:

```
For any continuous function f: ℝⁿ → ℝ and ε > 0,
there exists a single-hidden-layer network F such that:
|F(x) - f(x)| < ε for all x in the domain
```

### Implications

**Theoretical Power**:

- Neural networks are universal function approximators
- Guarantees existence of solution
- Does NOT guarantee we can find it via training

**Practical Limitations**:

1. **Width vs Depth**: Theorem proves width sufficiency, but depth is often more practical
2. **Number of Neurons**: May require exponentially many neurons
3. **Trainability**: Finding the weights is the hard part
4. **Generalization**: Approximating training data ≠ good generalization

### Why Deep Networks?

**Advantages of Depth**:

1. **Parameter Efficiency**: Can represent complex functions with fewer parameters
2. **Compositional Structure**: Matches structure of many problems
3. **Hierarchical Features**: Learns progressively abstract representations
4. **Empirical Success**: Deep networks consistently outperform shallow ones

## Network Capacity

### Expressiveness

**Definition**: The range of functions a network can represent

**Factors Affecting Capacity**:

1. Number of layers (depth)
2. Neurons per layer (width)
3. Type of activation functions
4. Total parameter count

**Trade-offs**:

- **Too little capacity**: Underfitting, cannot model complexity
- **Too much capacity**: Overfitting, poor generalization

### Degrees of Freedom

**Parameter Count**:

```
Total parameters = Σ(n⁽ˡ⁾ × n⁽ˡ⁻¹⁾ + n⁽ˡ⁾) for all layers l
```

**Rule of Thumb**:

- Need sufficient training examples relative to parameters
- Common ratio: 10-100x more examples than parameters (very problem-dependent)

## Forward Propagation Algorithm

### Specification

**Input**:

- Network parameters: {W⁽ˡ⁾, b⁽ˡ⁾} for l = 1 to L
- Input sample: x
- Activation functions: {f⁽ˡ⁾} for l = 1 to L

**Output**:

- Prediction: ŷ
- All intermediate activations (needed for backpropagation): {z⁽ˡ⁾, a⁽ˡ⁾}

**Algorithm**:

```
FORWARD_PROPAGATION(x, {W, b, f}):
    // Initialize
    a⁽⁰⁾ ← x
    
    // Process each layer
    FOR l = 1 TO L:
        // Linear transformation
        z⁽ˡ⁾ ← W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
        
        // Non-linear activation
        a⁽ˡ⁾ ← f⁽ˡ⁾(z⁽ˡ⁾)
        
        // Cache for backpropagation
        STORE(z⁽ˡ⁾, a⁽ˡ⁾)
    
    // Return prediction and cached values
    RETURN a⁽ᴸ⁾, {z⁽ˡ⁾, a⁽ˡ⁾}
```

### Batch Forward Propagation

**For m samples simultaneously**:

**Input**:

- Input matrix: X ∈ ℝⁿ⁽⁰⁾ ˣ ᵐ

**Algorithm**:

```
BATCH_FORWARD_PROPAGATION(X, {W, b, f}):
    A⁽⁰⁾ ← X
    
    FOR l = 1 TO L:
        Z⁽ˡ⁾ ← W⁽ˡ⁾ · A⁽ˡ⁻¹⁾ + b⁽ˡ⁾  (bias broadcasted)
        A⁽ˡ⁾ ← f⁽ˡ⁾(Z⁽ˡ⁾)            (element-wise)
        STORE(Z⁽ˡ⁾, A⁽ˡ⁾)
    
    RETURN A⁽ᴸ⁾, {Z⁽ˡ⁾, A⁽ˡ⁾}
```

## Network Topologies

### Fully Connected (Dense) Networks

**Definition**: Every neuron connected to every neuron in adjacent layers

**Characteristics**:

- Maximum connectivity
- Most parameters
- No assumptions about input structure

**Use Cases**:

- Tabular data
- Final layers of most architectures
- When input structure is not spatial

### Sparse Connectivity

**Motivation**: Reduce parameters, enforce structure

**Examples**:

- Convolutional layers (local connectivity)
- Graph neural networks (connectivity defined by graph)

**Benefits**:

- Fewer parameters
- Better generalization
- Inductive biases for specific domains

## Design Considerations

### Architecture Selection

**Factors to Consider**:

1. **Problem Type**: Classification, regression, generation, etc.
2. **Data Type**: Images, text, tabular, time-series
3. **Data Size**: More data supports larger networks
4. **Computational Budget**: Training and inference time
5. **Interpretability**: Simpler networks more interpretable

### Hyperparameters

**Network Structure**:

- Number of layers
- Neurons per layer
- Activation functions
- Connection patterns

**These are NOT learned during training**:

- Must be specified before training
- Often selected via:
  - Domain knowledge
  - Prior experience
  - Hyperparameter search
  - Neural architecture search (NAS)

## Information Flow

### Forward Flow

**Data Transformation**:

```
Raw Input → Features → Abstract Features → Prediction
```

**Each Layer**:

- Applies affine transformation (W·x + b)
- Applies non-linearity (activation function)
- Increases or decreases dimensionality

### Bottleneck Architecture

**Concept**: Compress information through narrow layer(s)

**Structure**:

```
Input (high dim) → Hidden (low dim) → Output (high dim)
```

**Applications**:

- Autoencoders
- Dimensionality reduction
- Feature learning

### Skip Connections

**Concept**: Connect non-adjacent layers

**Form**:

```
a⁽ˡ⁺ⁿ⁾ = f(W⁽ˡ⁺ⁿ⁾·a⁽ˡ⁺ⁿ⁻¹⁾ + a⁽ˡ⁾)
```

**Benefits**:

- Ease gradient flow
- Enable very deep networks
- Preserve information

**Architecture Examples**:

- ResNet (Residual Networks)
- DenseNet

## Training vs Inference

### Training Mode

**Purpose**: Learn optimal parameters

**Process**:

1. Forward propagation
2. Compute loss
3. Backward propagation (compute gradients)
4. Update parameters

**Requirements**:

- Training data
- Loss function
- Optimization algorithm
- Caching intermediate values

### Inference Mode

**Purpose**: Make predictions on new data

**Process**:

1. Forward propagation only
2. No gradient computation
3. No parameter updates

**Optimizations**:

- No need to cache intermediate values
- Can quantize weights
- Can prune unnecessary connections
- Can fuse operations

## Summary

### Key Concepts

1. **Perceptron**: Basic building block, creates linear decision boundary
2. **Multi-layer Networks**: Overcome perceptron limitations through composition
3. **Universal Approximation**: Networks can theoretically approximate any function
4. **Representation Learning**: Networks learn hierarchical features
5. **Forward Propagation**: Sequential computation through layers

### Design Principles

1. Depth enables efficient representation of complex functions
2. Width affects capacity at each level of abstraction
3. Activation functions provide non-linearity (essential!)
4. Architecture should match problem structure
5. More parameters require more data and compute

### Implementation Requirements

1. Efficient matrix operations
2. Support for various activation functions
3. Flexible architecture specification
4. Batch processing capability
5. Memory management for cached values

---

**Previous**: [Mathematical Foundations](./01-mathematical-foundations.md)  
**Next**: [Activation Functions](./03-activation-functions.md)
