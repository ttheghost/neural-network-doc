# Specifications vs. Tutorials: A Guide

## Two Different Approaches to Learning

This directory contains **two complementary sets of documentation**:

### 📋 Specifications (../\*.md files)

**Purpose**: Reference documentation
**Style**: Technical and comprehensive
**Audience**: People who need to look up specific details

**Located in**: `../01-mathematical-foundations.md`, `../02-neural-network-fundamentals.md`, etc.

### 📚 Tutorials (./\*.md files)

**Purpose**: Step-by-step learning
**Style**: Conversational and hands-on
**Audience**: People who want to build and understand deeply

**Located in**: `./01-building-tensor-class.md`, `./02-automatic-differentiation.md`, etc.

---

## Key Differences

| Aspect | Specifications | Tutorials |
|--------|---------------|-----------|
| **Goal** | Document what to build | Teach how and why to build |
| **Format** | Definitions, formulas, algorithms | Code walkthroughs, explanations, challenges |
| **Code** | Pseudocode and algorithms | Complete, runnable C++ code |
| **Length** | Concise and terse | Detailed and verbose |
| **Exercises** | None | Many challenges throughout |
| **Examples** | Minimal | Extensive, with full projects |

---

## When to Use Each

### Use Specifications When

- ✅ You need to look up a formula quickly
- ✅ You want a high-level overview of a concept
- ✅ You're implementing something and need the algorithm
- ✅ You want to see all options at a glance
- ✅ You're already experienced and just need a reminder

**Example**: "What's the derivative of the sigmoid function?"
→ Check `../03-activation-functions.md`

### Use Tutorials When

- ✅ You're learning a new concept from scratch
- ✅ You want to understand the implementation details
- ✅ You need to see working code examples
- ✅ You want to practice with exercises
- ✅ You're building the library step-by-step

**Example**: "How do I implement sigmoid with automatic differentiation?"
→ Follow `./02-automatic-differentiation.md`

---

## Mapping: Specifications → Tutorials

Here's how the specification files map to tutorial files:

| Specification File | Related Tutorial(s) | Coverage |
|-------------------|-------------------|----------|
| `01-mathematical-foundations.md` | Tutorial 01 | Tensors, matrix ops |
| `02-neural-network-fundamentals.md` | Tutorial 01, 02 | Network structure, forward pass |
| `03-activation-functions.md` | Tutorial 02 | Sigmoid, ReLU, tanh with gradients |
| `04-loss-functions.md` | Tutorial 03 | MSE, BCE, Cross-Entropy |
| `05-backpropagation.md` | Tutorial 02 | Autograd engine |
| `06-optimization-algorithms.md` | Tutorial 03 | SGD, Momentum, Adam |
| `07-regularization-techniques.md` | Tutorial 03, 08 | Weight decay, dropout |
| `08-network-architectures.md` | Tutorials 04, 05, 06 | CNNs, RNNs, Transformers |
| `09-training-strategies.md` | Tutorial 03 | Training loops, data loading |
| `10-api-design-principles.md` | All tutorials | Class design throughout |
| `11-library-architecture.md` | All tutorials | Overall structure |
| `12-performance-considerations.md` | Tutorials 07, 08 | GPU, optimization |

---

## Suggested Workflow

### For Complete Beginners

```
1. Read Tutorial README (./README.md)
2. Work through Tutorial 01 completely
3. Reference specifications as needed for theory
4. Continue with Tutorial 02, etc.
```

### For Experienced Developers

```
1. Skim through specifications for overview
2. Jump to relevant tutorial for implementation
3. Use specifications as quick reference
4. Complete tutorial challenges for practice
```

### For Quick Reference

```
1. Use specifications to find what you need
2. If you need more detail, check the tutorial
```

---

## Example: Learning About Loss Functions

### Specification Approach (`../04-loss-functions.md`)

```markdown
## Mean Squared Error

Formula: MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

Properties:
- Differentiable everywhere
- Heavily penalizes large errors
- Used for regression tasks

Gradient: ∂MSE/∂ŷ = 2(ŷ - y) / n
```

### Tutorial Approach (`./03-training-infrastructure.md`)

```cpp
// Let's build MSE from scratch!
class MSELoss : public Loss {
public:
    VariablePtr operator()(VariablePtr pred, VariablePtr target) override {
        // MSE = (pred - target)²
        // Why squared? Because we want to penalize large errors heavily
        // and ensure the loss is always positive
        
        auto diff = pred + (target * -1.0f);
        return pow(diff, 2.0f);  // Our autograd handles derivatives!
    }
    // ... more code with explanations
};

// Now let's test it:
auto pred = std::make_shared<Variable>(5.0f, true);
auto target = std::make_shared<Variable>(3.0f, false);
MSELoss loss;
auto l = loss(pred, target);
// l = (5-3)² = 4
```

**See the difference?**

- Specification: *What* MSE is
- Tutorial: *How* to implement it, *why* it works that way

---

## Philosophy

### Specifications Say

"Here is the algorithm for backpropagation."

### Tutorials Say

"Let's build backpropagation together! First, understand why we need it..."

Both are valuable, but for different purposes!

---

## Recommendations

### If You're Building the Library

1. **Start** with tutorials for step-by-step guidance
2. **Reference** specifications when you need algorithms
3. **Return** to specifications later as quick reference

### If You're Just Curious

1. **Skim** specifications to get the big picture
2. **Read** tutorials for specific topics that interest you
3. **Try** the challenges to test understanding

### If You're Stuck

1. **Check** specifications for correct algorithm
2. **Read** tutorial explanation for intuition
3. **Compare** your code with tutorial examples

---

## The Big Picture

Think of it like learning a language:

- **Specifications** = Dictionary and grammar reference
  - "Look, here's all the vocabulary and rules"
  
- **Tutorials** = Language course with a teacher
  - "Let me teach you how to actually speak and think in this language"

You need both!

---

## Contributing

### When to Update Specifications

- New algorithms or formulas
- Corrections to mathematical definitions
- Additional architecture patterns

### When to Update Tutorials

- Improved explanations
- More examples and challenges
- Better incremental learning steps
- Bug fixes in code

---

## Quick Navigation

### 📋 Want Theory?

→ Go to `../` for specification files

### 📚 Want to Build?

→ Stay in `./` for tutorial files

### 🎯 Want Both?

→ Use tutorials as primary, specifications as reference!

---

**Remember**: Specifications tell you *what* to build. Tutorials teach you *how* and *why*.

Use both to become a neural network master! 🚀
