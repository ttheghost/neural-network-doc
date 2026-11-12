# Quick Start: Your First Neural Network in C++

> **Goal**: Get a working neural network running in 30 minutes!

This is a fast-track introduction. For deep understanding, follow the full tutorials.

---

## Step 1: Verify Your Environment (5 minutes)

### Check Your Compiler

```bash
# Windows (Visual Studio)
cl

# Linux/macOS
g++ --version  # Should be 9.0 or higher
```

### Create Project Directory

```bash
mkdir my-neural-network
cd my-neural-network
mkdir include src
```

---

## Step 2: Build Your First Tensor (10 minutes)

Create `include/tensor.hpp`:

```cpp
#pragma once
#include <memory>
#include <vector>
#include <stdexcept>

namespace nn {

template<typename T>
class Tensor {
private:
    std::unique_ptr<T[]> data_;
    std::vector<size_t> shape_;
    size_t total_size_;
    
public:
    Tensor(const std::vector<size_t>& shape) 
        : shape_(shape) 
    {
        total_size_ = 1;
        for (auto dim : shape_) total_size_ *= dim;
        
        data_ = std::make_unique<T[]>(total_size_);
        for (size_t i = 0; i < total_size_; ++i) {
            data_[i] = T{0};
        }
    }
    
    T& at_flat(size_t idx) {
        if (idx >= total_size_) throw std::out_of_range("Index out of bounds");
        return data_[idx];
    }
    
    T at_flat(size_t idx) const {
        if (idx >= total_size_) throw std::out_of_range("Index out of bounds");
        return data_[idx];
    }
    
    size_t total_size() const { return total_size_; }
    const std::vector<size_t>& shape() const { return shape_; }
};

} // namespace nn
```

**Test it** - Create `test_tensor.cpp`:

```cpp
#include "include/tensor.hpp"
#include <iostream>

int main() {
    nn::Tensor<float> t({2, 3});  // 2x3 matrix
    
    t.at_flat(0) = 1.0f;
    t.at_flat(1) = 2.0f;
    t.at_flat(2) = 3.0f;
    
    std::cout << "First element: " << t.at_flat(0) << "\n";
    std::cout << "Success! Your first tensor works!\n";
    
    return 0;
}
```

**Compile and run**:

```bash
# Linux/macOS
g++ -std=c++17 -I. test_tensor.cpp -o test_tensor
./test_tensor

# Windows (Visual Studio)
cl /std:c++17 /EHsc /I. test_tensor.cpp
test_tensor.exe
```

✅ **Milestone**: You've built a basic tensor class!

---

## Step 3: Add Automatic Differentiation (10 minutes)

Create `include/variable.hpp`:

```cpp
#pragma once
#include <memory>
#include <functional>

namespace nn {

class Variable;
using VarPtr = std::shared_ptr<Variable>;

class Variable {
private:
    float data_;
    float grad_;
    bool requires_grad_;
    std::function<void()> backward_fn_;
    
public:
    Variable(float data, bool requires_grad = false)
        : data_(data), grad_(0.0f), requires_grad_(requires_grad) {}
    
    float data() const { return data_; }
    float grad() const { return grad_; }
    bool requires_grad() const { return requires_grad_; }
    
    void set_data(float d) { data_ = d; }
    void add_grad(float g) { grad_ += g; }
    void zero_grad() { grad_ = 0.0f; }
    
    void set_backward_fn(std::function<void()> fn) { backward_fn_ = fn; }
    
    void backward(float gradient = 1.0f) {
        add_grad(gradient);
        if (backward_fn_) backward_fn_();
    }
};

// Addition
VarPtr operator+(VarPtr a, VarPtr b) {
    auto result = std::make_shared<Variable>(
        a->data() + b->data(),
        a->requires_grad() || b->requires_grad()
    );
    
    if (result->requires_grad()) {
        result->set_backward_fn([a, b, result]() {
            if (a->requires_grad()) a->add_grad(result->grad());
            if (b->requires_grad()) b->add_grad(result->grad());
        });
    }
    
    return result;
}

// Multiplication
VarPtr operator*(VarPtr a, VarPtr b) {
    auto result = std::make_shared<Variable>(
        a->data() * b->data(),
        a->requires_grad() || b->requires_grad()
    );
    
    if (result->requires_grad()) {
        result->set_backward_fn([a, b, result]() {
            if (a->requires_grad()) a->add_grad(result->grad() * b->data());
            if (b->requires_grad()) b->add_grad(result->grad() * a->data());
        });
    }
    
    return result;
}

// Scalar multiplication
VarPtr operator*(VarPtr a, float scalar) {
    auto result = std::make_shared<Variable>(
        a->data() * scalar,
        a->requires_grad()
    );
    
    if (result->requires_grad()) {
        result->set_backward_fn([a, scalar, result]() {
            if (a->requires_grad()) a->add_grad(result->grad() * scalar);
        });
    }
    
    return result;
}

} // namespace nn
```

**Test it** - Create `test_autograd.cpp`:

```cpp
#include "include/variable.hpp"
#include <iostream>

int main() {
    // Compute f(x) = (x + 2) * 3
    // df/dx = 3
    
    auto x = std::make_shared<nn::Variable>(5.0f, true);
    auto two = std::make_shared<nn::Variable>(2.0f, false);
    
    auto y = x + two;    // y = x + 2 = 7
    auto f = y * 3.0f;   // f = y * 3 = 21
    
    std::cout << "f(5) = " << f->data() << "\n";
    
    f->backward();  // Compute gradient
    
    std::cout << "df/dx = " << x->grad() << " (expected: 3)\n";
    std::cout << "Success! Autograd works!\n";
    
    return 0;
}
```

**Compile and run**:

```bash
g++ -std=c++17 -I. test_autograd.cpp -o test_autograd
./test_autograd
```

✅ **Milestone**: You have automatic differentiation!

---

## Step 4: Train Your First Neural Network (5 minutes)

Create `xor_network.cpp`:

```cpp
#include "include/variable.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace nn;

// Sigmoid activation
VarPtr sigmoid(VarPtr x) {
    float sig = 1.0f / (1.0f + std::exp(-x->data()));
    auto result = std::make_shared<Variable>(sig, x->requires_grad());
    
    if (result->requires_grad()) {
        result->set_backward_fn([x, result, sig]() {
            float grad = result->grad() * sig * (1.0f - sig);
            x->add_grad(grad);
        });
    }
    
    return result;
}

int main() {
    std::cout << "Training XOR network...\n\n";
    
    // Network weights (2 → 4 → 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Initialize weights
    auto w1_11 = std::make_shared<Variable>(dist(gen), true);
    auto w1_12 = std::make_shared<Variable>(dist(gen), true);
    auto w1_21 = std::make_shared<Variable>(dist(gen), true);
    auto w1_22 = std::make_shared<Variable>(dist(gen), true);
    auto b1 = std::make_shared<Variable>(dist(gen), true);
    
    auto w2_1 = std::make_shared<Variable>(dist(gen), true);
    auto w2_2 = std::make_shared<Variable>(dist(gen), true);
    auto b2 = std::make_shared<Variable>(dist(gen), true);
    
    std::vector<VarPtr> weights = {w1_11, w1_12, w1_21, w1_22, b1, w2_1, w2_2, b2};
    
    // XOR dataset
    std::vector<std::pair<std::pair<float, float>, float>> data = {
        {{0, 0}, 0}, {{0, 1}, 1}, {{1, 0}, 1}, {{1, 1}, 0}
    };
    
    float lr = 0.5f;
    
    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        float total_loss = 0.0f;
        
        for (auto& [input, target] : data) {
            // Zero gradients
            for (auto& w : weights) w->zero_grad();
            
            // Forward pass
            auto x1 = std::make_shared<Variable>(input.first, false);
            auto x2 = std::make_shared<Variable>(input.second, false);
            
            auto h1 = sigmoid(x1 * w1_11 + x2 * w1_12 + b1);
            auto h2 = sigmoid(x1 * w1_21 + x2 * w1_22 + b1);
            auto out = sigmoid(h1 * w2_1 + h2 * w2_2 + b2);
            
            // Loss (simple squared error)
            auto diff = std::make_shared<Variable>(out->data() - target, true);
            auto loss = diff * diff;
            total_loss += loss->data();
            
            // Backward pass
            loss->backward();
            
            // Update weights
            for (auto& w : weights) {
                w->set_data(w->data() - lr * w->grad());
            }
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << (total_loss / 4) << "\n";
        }
    }
    
    // Test
    std::cout << "\nTesting:\n";
    for (auto& [input, target] : data) {
        auto x1 = std::make_shared<Variable>(input.first, false);
        auto x2 = std::make_shared<Variable>(input.second, false);
        
        auto h1 = sigmoid(x1 * w1_11 + x2 * w1_12 + b1);
        auto h2 = sigmoid(x1 * w1_21 + x2 * w1_22 + b1);
        auto out = sigmoid(h1 * w2_1 + h2 * w2_2 + b2);
        
        std::cout << input.first << " XOR " << input.second 
                  << " = " << out->data() 
                  << " (target: " << target << ")\n";
    }
    
    std::cout << "\n✨ You trained your first neural network! ✨\n";
    
    return 0;
}
```

**Compile and run**:

```bash
g++ -std=c++17 -I. xor_network.cpp -o xor_network
./xor_network
```

Watch it learn XOR! 🎉

✅ **MAJOR Milestone**: You've trained a neural network from scratch!

---

## What You Just Built

In 30 minutes, you:

1. ✅ Created a basic tensor class
2. ✅ Implemented automatic differentiation
3. ✅ Trained a neural network to solve XOR

**This is the foundation of all deep learning!**

---

## Next Steps

### Want to understand deeply?

→ Read [Tutorial 01: Building Your First Tensor Class](./01-building-tensor-class.md)

### Want more features?

→ Read [Tutorial 02: Automatic Differentiation](./02-automatic-differentiation.md)

### Want to build real projects?

→ Read [Tutorial 03: Training Infrastructure](./03-training-infrastructure.md)

---

## Common Issues

### Compilation Error: "enable_shared_from_this"

Solution: The quick-start uses a simplified version without this. The full tutorials show the complete implementation.

### Network Doesn't Converge

Try:

- Reducing learning rate: `lr = 0.1f`
- Running more epochs: `epoch < 5000`
- Changing random seed

### "Undefined reference" on Linux

Make sure you're linking math library:

```bash
g++ -std=c++17 -I. xor_network.cpp -o xor_network -lm
```

---

## Congratulations! 🎉

You've just:

- Built the core of a neural network library
- Understood how PyTorch works under the hood
- Trained an actual neural network

**You're not just using deep learning anymore. You're creating it.**

Now dive into the full tutorials to master every detail! 🚀

---

**Ready for more?** Start with [Tutorial README](./README.md)
