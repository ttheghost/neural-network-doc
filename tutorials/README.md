# Neural Network From Scratch: A Hands-On Tutorial Series

> **Welcome!** This is a comprehensive, step-by-step guide to building a neural network library in C++ from the ground up. Inspired by [craftinginterpreters.com](https://craftinginterpreters.com), this series teaches you not just *what* to build, but *why* and *how*.

## 📚 Philosophy

Unlike typical specifications or API documentation, these tutorials are designed like a **university course**:

- **Incremental Learning**: Each concept builds on the previous one
- **Hands-On Practice**: You'll write every line of code yourself
- **Deep Understanding**: We explain the "why" behind every decision
- **Practical Challenges**: Exercises reinforce your learning
- **Real-World Examples**: Every tutorial ends with a working project

## 🎯 Who Is This For?

This series is perfect if you:

- Want to understand how PyTorch/TensorFlow work under the hood
- Are comfortable with C++ (know templates, pointers, RAII)
- Love building things from scratch
- Want to deeply understand neural networks, not just use them

**Prerequisites**:

- C++17 or later
- Basic linear algebra (matrix multiplication, derivatives)
- Curiosity and determination 🚀

---

## 📖 Tutorial Series

### Foundation (Start Here!)

#### [Tutorial 01: Building Your First Tensor Class](./01-building-tensor-class.md)

**Time**: 4-6 hours | **Difficulty**: ⭐⭐⭐

Build the foundation of any deep learning library: a multi-dimensional tensor class.

**What You'll Learn**:

- Dynamic multi-dimensional arrays
- Memory management with modern C++ (unique_ptr, move semantics)
- Shape, strides, and indexing
- Element-wise operations
- Matrix multiplication
- Broadcasting (advanced)

**What You'll Build**:

- A complete `Tensor<T>` class
- Mathematical operations (+, -, *, /)
- Matrix multiplication implementation
- **Final Project**: Neural network inference for XOR problem

**Key Takeaway**: Understanding tensors is understanding deep learning.

---

#### [Tutorial 02: Building an Automatic Differentiation Engine](./02-automatic-differentiation.md)

**Time**: 5-7 hours | **Difficulty**: ⭐⭐⭐⭐

Build the magic that makes neural networks trainable: automatic differentiation (autograd).

**What You'll Learn**:

- Computational graphs
- Forward and backward propagation
- The chain rule in practice
- Topological sorting
- Activation functions
- Memory management in computational graphs

**What You'll Build**:

- A complete `Variable` class with gradient tracking
- Operations: +, -, *, /, pow
- Activation functions: sigmoid, ReLU, tanh
- **Final Project**: Train a 2-layer network to solve XOR

**Key Takeaway**: Autodiff turns calculus into graph traversal.

---

#### [Tutorial 03: Building Training Infrastructure](./03-training-infrastructure.md)

**Time**: 5-7 hours | **Difficulty**: ⭐⭐⭐⭐

Build the complete training pipeline: optimizers, loss functions, and data loaders.

**What You'll Learn**:

- Loss functions (MSE, Cross-Entropy)
- Optimization algorithms (SGD, Momentum, Adam)
- Data loading and batching
- Training loops
- Metrics and evaluation
- Regularization techniques

**What You'll Build**:

- Loss functions: MSE, BCE, Cross-Entropy
- Optimizers: SGD, SGD+Momentum, Adam
- DataLoader with batching
- Complete `Trainer` class
- **Final Project**: Train a regression model on synthetic data

**Key Takeaway**: Training is 90% infrastructure, 10% forward pass.

---

### Advanced Topics (Coming Soon)

#### Tutorial 04: Convolutional Neural Networks (CNNs)

**Time**: 6-8 hours | **Difficulty**: ⭐⭐⭐⭐

Build the architecture that revolutionized computer vision.

**What You'll Learn**:

- Convolutional layers
- Pooling operations
- Local connectivity and weight sharing
- Translation invariance
- Backpropagation through convolutions

**What You'll Build**:

- Conv2D layer
- MaxPool2D layer
- Complete CNN for MNIST
- **Final Project**: Classify handwritten digits (MNIST)

---

#### Tutorial 05: Recurrent Neural Networks (RNNs)

**Time**: 6-8 hours | **Difficulty**: ⭐⭐⭐⭐⭐

Build networks that process sequences and remember the past.

**What You'll Learn**:

- Recurrent connections
- Unrolling through time
- LSTM and GRU architectures
- Vanishing/exploding gradients
- Sequence-to-sequence models

**What You'll Build**:

- Basic RNN cell
- LSTM cell
- Sequence classifier
- **Final Project**: Character-level text generation

---

#### Tutorial 06: Attention Mechanisms and Transformers

**Time**: 8-10 hours | **Difficulty**: ⭐⭐⭐⭐⭐

Build the architecture behind GPT, BERT, and modern LLMs.

**What You'll Learn**:

- Attention mechanism
- Multi-head attention
- Positional encoding
- Transformer architecture
- Self-attention vs cross-attention

**What You'll Build**:

- Attention layer
- Multi-head attention
- Transformer block
- **Final Project**: Simple sequence-to-sequence translation

---

### Performance & Production (Advanced)

#### Tutorial 07: GPU Acceleration with CUDA

**Time**: 10-12 hours | **Difficulty**: ⭐⭐⭐⭐⭐

Make your library actually fast with GPU acceleration.

**What You'll Learn**:

- CUDA basics
- Memory management (host vs device)
- Kernel programming
- cuBLAS and cuDNN
- Memory transfer optimization

**What You'll Build**:

- GPU tensor operations
- CUDA kernels for common operations
- Benchmark: CPU vs GPU speedup

---

#### Tutorial 08: Production Optimization

**Time**: 8-10 hours | **Difficulty**: ⭐⭐⭐⭐⭐

Transform your educational library into a production-ready framework.

**What You'll Learn**:

- Operator fusion
- Memory pooling
- Graph optimization
- JIT compilation
- Quantization
- Model serialization

**What You'll Build**:

- Graph optimizer
- Memory allocator
- Model save/load
- Benchmark suite

---

## 🗺️ Learning Paths

### Path 1: Understanding Deep Learning (Recommended for Beginners)

```
Tutorial 01 → Tutorial 02 → Tutorial 03 → Tutorial 04
```

This path gives you a solid foundation in neural networks.

### Path 2: Natural Language Processing

```
Tutorials 01-03 → Tutorial 05 → Tutorial 06
```

Perfect for those interested in text and language models.

### Path 3: High-Performance Computing

```
Tutorials 01-03 → Tutorial 07 → Tutorial 08
```

For those who want to build fast, production-ready systems.

### Path 4: Complete Mastery

```
Tutorials 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

The full journey from zero to production-ready deep learning library.

---

## 💡 How to Use These Tutorials

### 1. **Type Everything Yourself**

Don't copy-paste! Typing code forces you to think about every line.

### 2. **Do the Challenges**

Each tutorial has exercises (🎯). These are crucial for understanding.

### 3. **Experiment**

Change parameters, break things, fix them. That's how you learn.

### 4. **Build the Final Projects**

Each tutorial ends with a capstone project. Complete it before moving on.

### 5. **Take Notes**

Keep a learning journal. Write down "aha!" moments and confusing parts.

---

## 🔍 Key Concepts by Tutorial

| Tutorial | Key Concepts |
|----------|-------------|
| 01 | Tensors, Memory Management, Linear Algebra |
| 02 | Computational Graphs, Chain Rule, Backpropagation |
| 03 | Optimization, Loss Functions, Training Loops |
| 04 | Convolution, Local Connectivity, Image Processing |
| 05 | Recurrence, Memory, Sequence Modeling |
| 06 | Attention, Transformers, Modern NLP |
| 07 | GPU Programming, CUDA, Parallelization |
| 08 | Optimization, Production, Real-World Deployment |

---

## 📊 Estimated Time Commitment

- **Foundation (Tutorials 1-3)**: ~15-20 hours
- **Advanced (Tutorials 4-6)**: ~20-25 hours
- **Performance (Tutorials 7-8)**: ~18-22 hours

**Total**: ~55-70 hours for complete mastery

**Recommendation**: Spend 2-3 hours per session, with breaks. Quality > speed.

---

## 🛠️ Setting Up Your Environment

### Compiler Requirements

- **Windows**: Visual Studio 2019+ or MinGW-w64 with g++ 9+
- **Linux**: g++ 9+ or clang++ 10+
- **macOS**: Xcode 12+ or clang++ 10+

### Recommended Tools

- **Editor**: VSCode with C++ extensions
- **Build System**: CMake 3.15+
- **Debugger**: gdb or lldb
- **Profiler**: perf (Linux), Instruments (macOS), VS Profiler (Windows)

### Optional Dependencies

- **Testing**: Google Test
- **Benchmarking**: Google Benchmark
- **BLAS**: OpenBLAS or Intel MKL (for optimized matrix ops)
- **GPU**: CUDA Toolkit 11.0+ (for Tutorial 07)

---

## 📝 Project Structure

As you progress through the tutorials, you'll build this directory structure:

```
neural-network/
├── include/
│   ├── tensor.hpp         # Tutorial 01
│   ├── autograd.hpp       # Tutorial 02
│   ├── optimizer.hpp      # Tutorial 03
│   ├── loss.hpp           # Tutorial 03
│   ├── dataset.hpp        # Tutorial 03
│   ├── layers.hpp         # Tutorial 04
│   └── ...
├── src/
│   ├── tensor.cpp
│   ├── layers.cpp
│   └── ...
├── tests/
│   ├── test_tensor.cpp
│   ├── test_autograd.cpp
│   └── ...
├── examples/
│   ├── xor_problem.cpp
│   ├── mnist_cnn.cpp
│   └── ...
└── CMakeLists.txt
```

---

## 🎓 Learning Resources

### Books

- **"Deep Learning" by Goodfellow, Bengio, Courville** - The Bible
- **"Neural Networks and Deep Learning" by Michael Nielsen** - Free online
- **"Effective Modern C++" by Scott Meyers** - For C++ best practices

### Papers (Read After Completing Tutorials)

- **Backpropagation**: Rumelhart et al. (1986)
- **Adam Optimizer**: Kingma & Ba (2014)
- **ResNet**: He et al. (2015)
- **Attention Is All You Need**: Vaswani et al. (2017)

### Online Resources

- [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/) - How PyTorch works
- [Andrej Karpathy's Blog](https://karpathy.github.io/) - Deep learning insights
- [Chris Olah's Blog](https://colah.github.io/) - Visual explanations

---

## 🤔 Common Questions

### Q: Do I need a GPU?

**A**: Not for Tutorials 1-6! CPU is fine for learning. Tutorial 7 requires a NVIDIA GPU.

### Q: Can I use C++20/23 features?

**A**: Absolutely! We use C++17 as a baseline, but modern features are encouraged.

### Q: Should I read the specification docs too?

**A**: The tutorials are self-contained. Spec docs are reference material, not learning material.

### Q: How is this different from using PyTorch?

**A**: PyTorch teaches you *what*. These tutorials teach you *how* and *why*.

### Q: Can I contribute improvements?

**A**: Yes! If you find bugs or have suggestions, contributions are welcome.

---

## 🏆 Milestones

Track your progress:

- [ ] **Milestone 1**: Built a working tensor class
- [ ] **Milestone 2**: Computed first automatic gradient
- [ ] **Milestone 3**: Trained first neural network (XOR)
- [ ] **Milestone 4**: Built complete training infrastructure
- [ ] **Milestone 5**: Classified MNIST digits with CNN
- [ ] **Milestone 6**: Generated text with RNN
- [ ] **Milestone 7**: Built attention mechanism
- [ ] **Milestone 8**: Ran code on GPU
- [ ] **Milestone 9**: Optimized for production

**Share your progress** on Twitter/LinkedIn with #BuildingNNFromScratch!

---

## 🎯 Final Goal

By the end of this series, you will have:

1. **Understanding**: Deep intuition for how neural networks actually work
2. **Skills**: Ability to implement any neural network architecture from papers
3. **Codebase**: A production-ready (or close to it) neural network library
4. **Confidence**: To contribute to frameworks like PyTorch or TensorFlow

---

## 🚀 Let's Begin

Ready to start your journey? Head to [Tutorial 01: Building Your First Tensor Class](./01-building-tensor-class.md)!

Remember: **The best way to learn is to build.** Let's build something amazing together! 💪

---

## 📬 Feedback and Support

Found a bug? Have a question? Want to share what you built?

- **Issues**: Open an issue in the repository
- **Discussions**: Use GitHub Discussions
- **Email**: [your-email@example.com]

---

*"What I cannot create, I do not understand."* - Richard Feynman

Let's create, and thereby understand, neural networks from scratch.

**Happy Building!** 🎉
