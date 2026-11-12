# Network Architectures

## Overview

This document describes specialized neural network architectures beyond basic feedforward networks. Each architecture is designed to handle specific types of data and tasks, with inductive biases that make learning more efficient.

## Feedforward Networks (Fully Connected)

### Architecture

**Structure**: Sequential layers of neurons, each fully connected to the next

**Mathematical Form**:

```
For each layer l:
z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = f(z⁽ˡ⁾)
```

### Characteristics

**Connectivity**:

- Every neuron connected to all neurons in next layer
- No assumptions about input structure
- Maximum expressiveness, maximum parameters

**Parameters**: O(n² × L) where n is layer width, L is depth

### Use Cases

- Tabular data
- Fixed-size inputs without spatial/temporal structure
- Final layers in most architectures
- Simple baseline models

### Limitations

- No spatial/temporal awareness
- Many parameters for high-dimensional inputs
- Doesn't scale well to images or sequences

## Convolutional Neural Networks (CNNs)

### Motivation

**Problems with Fully Connected for Images**:

- Images as vectors lose spatial structure
- Too many parameters (e.g., 224×224×3 image → 150K inputs)
- Doesn't leverage translation invariance

**Key Insights**:

1. Local connectivity: nearby pixels related
2. Translation invariance: features useful anywhere
3. Hierarchical patterns: edges → textures → parts → objects

### Convolutional Layer

**Operation**: Apply same filter across spatial dimensions

**Mathematical Form**:

```
For output position (i,j):
z_{i,j,k} = Σₘ Σₙ Σ_c W_{m,n,c,k} · x_{i+m, j+n, c} + b_k

where:
- (i,j): spatial position in output
- k: output channel (filter index)
- (m,n): positions within filter
- c: input channel
```

**Components**:

- **Kernel/Filter**: Small weight matrix (e.g., 3×3, 5×5)
- **Stride**: Step size when sliding filter
- **Padding**: Border handling (valid, same, full)
- **Channels**: Multiple filters → multiple output channels

**Hyperparameters**:

- Filter size: 1×1, 3×3, 5×5, 7×7
- Number of filters: 32, 64, 128, 256, etc.
- Stride: 1, 2 (stride 2 downsamples)
- Padding: 0 (valid), (K-1)/2 (same), K-1 (full)

### Parameter Sharing

**Advantage**:

```
Fully connected: n_input × n_output parameters
Convolutional: K × K × C_in × C_out parameters

K: kernel size (e.g., 3)
C_in: input channels
C_out: output channels

Dramatically fewer parameters!
```

**Example**:

```
200×200 RGB image → 128 feature maps

Fully connected:
(200×200×3) × (200×200×128) = 6.14 trillion parameters!

Convolutional (3×3 kernel):
(3×3×3) × 128 = 3,456 parameters

~1,000,000x fewer parameters
```

### Pooling Layers

**Purpose**: Downsample spatial dimensions

**Max Pooling**:

```
For each pool region:
output = max(inputs in region)

Example 2×2 max pooling:
[1 3]
[2 4] → 4  (maximum value)
```

**Average Pooling**:

```
For each pool region:
output = mean(inputs in region)
```

**Properties**:

- Reduces spatial size
- Reduces parameters in later layers
- Provides translation invariance
- No learnable parameters

**Common Configurations**:

- 2×2 pooling, stride 2 (halves dimensions)
- 3×3 pooling, stride 2

### Global Pooling

**Global Average Pooling (GAP)**:

```
For each channel:
output = average of entire spatial dimension

Input: H × W × C
Output: 1 × 1 × C (or just C)
```

**Advantages**:

- Drastically reduces parameters vs fully connected
- No spatial information → more robust
- Used before final classification layer

### Typical CNN Architecture

**Pattern**:

```
Input Image
↓
[Conv → Activation → (BatchNorm) → (Pooling)] × N  (feature extraction)
↓
[Fully Connected → Activation → (Dropout)] × M  (classification)
↓
Output (softmax for classification)
```

**Common Configurations**:

- Multiple conv layers before pooling
- Spatial size decreases, channels increase
- Example: 224×224×3 → 112×112×64 → 56×56×128 → 28×28×256

### Classic CNN Architectures

**LeNet-5** (1998):

```
Input → Conv(5×5) → Pool → Conv(5×5) → Pool → FC → FC → Output
- First successful CNN
- Handwritten digit recognition
```

**AlexNet** (2012):

```
5 Conv layers + 3 FC layers
- ReLU activation
- Dropout
- GPU training
- Won ImageNet 2012
```

**VGGNet** (2014):

```
Many small (3×3) convolutions
16-19 layers
Simple, uniform architecture
```

**ResNet** (2015):

```
Skip connections (residual connections)
Enables very deep networks (50, 101, 152 layers)
Identity shortcuts: a⁽ˡ⁺¹⁾ = f(a⁽ˡ⁾) + a⁽ˡ⁾
```

### Receptive Field

**Definition**: Region of input that affects a particular neuron

**Calculation**:

```
Layer 1 (3×3 kernel): 3×3 receptive field
Layer 2 (3×3 kernel): 5×5 receptive field  (overlapping patches)
Layer 3 (3×3 kernel): 7×7 receptive field

Each layer increases receptive field
```

**Design Consideration**: Deep network needed for large receptive fields

### Dilated Convolutions

**Motivation**: Increase receptive field without increasing parameters

**Operation**: Introduce gaps in filter

**Dilation Rate**:

- Rate 1: Standard convolution
- Rate 2: One pixel gap between filter positions
- Rate 4: Three pixel gap

**Benefits**:

- Larger receptive field
- Same computational cost
- Used in segmentation, audio models

## Recurrent Neural Networks (RNNs)

### Motivation

**Sequential Data Challenges**:

- Variable length inputs/outputs
- Temporal dependencies
- Order matters
- Examples: Text, time series, audio, video

**Limitations of Feedforward**:

- Fixed input size
- No memory of previous inputs
- Each input processed independently

### Basic RNN

**Architecture**: Hidden state passed through time

**Mathematical Form**:

```
At time step t:
h_t = f(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y

where:
- x_t: input at time t
- h_t: hidden state at time t
- y_t: output at time t
- h_0: initial hidden state (often zeros)
```

**Parameters**: Shared across all time steps!

- W_xh: input to hidden
- W_hh: hidden to hidden (recurrent)
- W_hy: hidden to output

### Unrolling Through Time

**Conceptual View**:

```
x_1 → [RNN] → h_1 → y_1
       ↓
x_2 → [RNN] → h_2 → y_2
       ↓
x_3 → [RNN] → h_3 → y_3
```

Same network, applied at each step

### Input/Output Configurations

**One-to-Many**:

```
Input: Single (e.g., image)
Output: Sequence (e.g., caption)

Example: Image captioning
```

**Many-to-One**:

```
Input: Sequence (e.g., sentence)
Output: Single (e.g., sentiment)

Example: Text classification
```

**Many-to-Many** (same length):

```
Input: Sequence
Output: Sequence (same length)

Example: Video frame labeling
```

**Many-to-Many** (different lengths):

```
Input: Sequence
Output: Sequence (different length)

Example: Machine translation (encoder-decoder)
```

### Backpropagation Through Time (BPTT)

**Algorithm**:

1. Unroll network through time
2. Forward pass: compute all h_t, y_t
3. Backward pass: compute gradients from end to start
4. Update parameters (shared across time)

**Truncated BPTT**:

- Limit backpropagation to k time steps
- Reduces memory and computation
- Prevents vanishing gradients over very long sequences

### Vanishing/Exploding Gradients

**Problem**: Gradients multiply through time

**Vanishing**:

```
If |∂h_t/∂h_{t-1}| < 1, gradients shrink exponentially
Long-term dependencies not learned
```

**Exploding**:

```
If |∂h_t/∂h_{t-1}| > 1, gradients grow exponentially
Training unstable
```

**Solutions**:

- Gradient clipping (for exploding)
- LSTM/GRU cells (for vanishing)
- Skip connections

## Long Short-Term Memory (LSTM)

### Motivation

**RNN Limitations**:

- Vanishing gradient problem
- Difficulty learning long-term dependencies
- Simple additive update insufficient

### LSTM Cell

**Components**:

1. **Cell state** (c_t): Long-term memory
2. **Hidden state** (h_t): Short-term memory / output
3. **Gates**: Control information flow

**Gates**:

```
Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

Candidate: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
```

**State Updates**:

```
Cell state:   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
Hidden state: h_t = o_t ⊙ tanh(c_t)

where ⊙ is element-wise multiplication
```

**Intuition**:

- **Forget gate**: What to discard from cell state
- **Input gate**: What new information to add
- **Output gate**: What to output from cell state

### Why LSTM Works

**Gradient Flow**:

- Cell state is "highway" for gradients
- Additive update (not multiplicative like vanilla RNN)
- Gates learned to preserve or reset information
- Can learn to remember very long sequences

**Flexible Memory**:

- Can learn to store information for many steps
- Can learn to clear memory when needed
- Gates adapt to data

### GRU (Gated Recurrent Unit)

**Simpler Alternative to LSTM**:

**Gates**:

```
Reset gate:  r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
Update gate: z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
```

**Updates**:

```
Candidate: h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
Hidden:    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Differences from LSTM**:

- Fewer parameters (no separate cell state)
- Often comparable performance
- Faster to train

## Bidirectional RNNs

### Concept

**Motivation**: Future context is informative

**Architecture**: Two RNNs, one forward, one backward

**Forward RNN**:

```
h→_t = f(W→ · h→_{t-1} + W_x→ · x_t)
```

**Backward RNN**:

```
h←_t = f(W← · h←_{t+1} + W_x← · x_t)
```

**Combined Output**:

```
h_t = [h→_t; h←_t]  (concatenation)
or
h_t = h→_t + h←_t  (addition)
```

### Use Cases

- Text understanding (context from both directions)
- Speech recognition
- Sequence labeling (NER, POS tagging)

**Limitation**: Requires entire sequence (not suitable for online/real-time)

## Encoder-Decoder Architecture

### Concept

**Sequence-to-Sequence**: Map input sequence to output sequence (different lengths)

**Structure**:

```
Encoder: Input sequence → Fixed-size context vector
Decoder: Context vector → Output sequence
```

### Encoder

**Role**: Compress input sequence into context vector

**Implementation**:

```
For input x_1, x_2, ..., x_T:

Process with RNN/LSTM:
h_t = RNN(x_t, h_{t-1})

Context vector: c = h_T (final hidden state)
Or: c = some_function(h_1, h_2, ..., h_T)
```

### Decoder

**Role**: Generate output sequence from context

**Implementation**:

```
Initialize: h_0 = c (context from encoder)

For each output step:
h_t = RNN(y_{t-1}, h_t)  // y_{t-1}: previous output
y_t = softmax(W · h_t)    // predict next token
```

**Teacher Forcing** (training):

- Use true previous output y_{t-1} (not predicted)
- Faster training, more stable
- At test time, use predicted outputs

### Attention Mechanism

**Problem**: Fixed context vector is bottleneck

**Solution**: Attend to all encoder states

**Attention Computation**:

```
For decoder state h_t:

// Compute attention scores
e_{t,i} = score(h_t, h_encoder_i)

// Normalize to weights
α_{t,i} = exp(e_{t,i}) / Σⱼ exp(e_{t,j})

// Weighted sum of encoder states
context_t = Σᵢ α_{t,i} · h_encoder_i

// Use in decoder
h_t = RNN([y_{t-1}; context_t], h_{t-1})
```

**Score Functions**:

- Dot product: h_t^T · h_encoder_i
- General: h_t^T · W · h_encoder_i
- Concat: W · tanh(W_1·h_t + W_2·h_encoder_i)

**Benefits**:

- No fixed bottleneck
- Decoder can "look" at relevant parts
- Interpretable (attention weights show focus)

## Transformer Architecture

### Motivation

**RNN Limitations**:

- Sequential processing (hard to parallelize)
- Long-range dependencies still challenging
- Slow training on long sequences

**Transformer**: Relies entirely on attention, no recurrence

### Self-Attention

**Concept**: Attend to all positions in same sequence

**Mathematical Form**:

```
Query: Q = X · W_Q
Key:   K = X · W_K
Value: V = X · W_V

Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

where:
- X: input sequence (n × d_model)
- d_k: key dimension
- √d_k: scaling factor
```

**Intuition**:

- Q: What I'm looking for
- K: What I offer
- V: What I actually contain
- Attention weights: Similarity between queries and keys

### Multi-Head Attention

**Concept**: Multiple attention in parallel (different subspaces)

**Implementation**:

```
For each head h:
  Q_h = X · W_Q^h
  K_h = X · W_K^h
  V_h = X · W_V^h
  
  head_h = Attention(Q_h, K_h, V_h)

MultiHead = Concat(head_1, ..., head_H) · W_O
```

**Benefits**:

- Attend to different aspects simultaneously
- More expressive
- 8-16 heads typical

### Positional Encoding

**Problem**: Self-attention is position-agnostic (permutation invariant)

**Solution**: Add positional information

**Sinusoidal Encoding**:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Added to input embeddings
```

**Learned Positional Embeddings**: Also common

### Transformer Block

**Structure**:

```
Input
↓
Multi-Head Self-Attention
↓
Add & Normalize (residual connection)
↓
Feed-Forward Network (2 FC layers)
↓
Add & Normalize (residual connection)
↓
Output
```

**Feed-Forward**:

```
FFN(x) = ReLU(x · W_1 + b_1) · W_2 + b_2

Applied position-wise (same for all positions)
```

### Full Transformer

**Encoder**:

- Stack of N Transformer blocks (N=6 typical)
- Self-attention on input sequence

**Decoder**:

- Stack of N Transformer blocks
- Self-attention on output sequence (masked)
- Cross-attention to encoder output
- Generates output autoregressively

### Use Cases

- Machine translation (original use)
- Text generation (GPT)
- Text understanding (BERT)
- Vision (Vision Transformer)
- Multi-modal models

**Advantages**:

- Highly parallelizable
- Long-range dependencies
- State-of-the-art results

**Disadvantages**:

- Computational cost O(n²) for sequence length n
- Large memory requirements
- Requires lots of data

## Architecture Selection Guide

### Data Type → Architecture

**Images**: CNN

- AlexNet, VGG, ResNet, EfficientNet
- Leverages spatial structure

**Text**: Transformer or RNN/LSTM

- Short sequences: CNN also works
- Modern: Transformer (BERT, GPT)
- Resource-constrained: LSTM/GRU

**Time Series**: RNN/LSTM or 1D CNN

- LSTM for long-term dependencies
- 1D CNN for fast inference

**Tabular**: Feedforward (Dense)

- No spatial/temporal structure
- Sometimes: Gradient boosting outperforms

**Multimodal**: Depends on modalities

- Often: Specialized encoders + fusion

### Task → Architecture Considerations

**Classification**:

- CNN (images)
- Transformer (text)
- Final global pooling + softmax

**Segmentation**:

- U-Net architecture (encoder-decoder with skip connections)
- Fully convolutional

**Object Detection**:

- R-CNN family, YOLO, etc.
- Specialized architectures

**Sequence-to-Sequence**:

- Transformer (modern)
- LSTM encoder-decoder (classic)

## Summary

### Key Architecture Families

1. **Feedforward**: General purpose, tabular data
2. **CNN**: Images, spatial data
3. **RNN/LSTM**: Sequences, temporal data
4. **Transformer**: Sequences, state-of-the-art for text

### Design Principles

- **Match architecture to data structure**
- **Use proven architectures as starting points**
- **Consider computational constraints**
- **Deeper is often better** (with residual connections)
- **Attention is powerful** (but expensive)

### Implementation Priorities

**Essential**:

- Feedforward (dense) layers
- Convolutional layers
- Pooling layers

**Important**:

- LSTM cells
- Batch normalization
- Residual connections

**Advanced**:

- Attention mechanisms
- Transformer blocks
- Specialized architectures

---

**Previous**: [Regularization Techniques](./07-regularization-techniques.md)  
**Next**: [Training Strategies](./09-training-strategies.md)
