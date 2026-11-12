# Tutorial 02: Building an Automatic Differentiation Engine

> **Learning Goal**: By the end of this tutorial, you'll understand how automatic differentiation works and build your own autograd system from scratch. This is the secret sauce that makes modern deep learning possible!

## The Big Picture: Why Do We Need This?

Imagine you have a neural network with **millions** of parameters. To train it, you need to compute the gradient of the loss function with respect to **every single parameter**.

**Doing this by hand?** Impossible.

**Using numerical differentiation?** Too slow and imprecise.

**The solution?** **Automatic differentiation (autodiff)** - a technique that computes derivatives automatically with machine precision. This is what PyTorch, TensorFlow, and JAX use under the hood.

### What Makes Autodiff Special?

Consider this computation:

```cpp
y = (x * 2) + 3
z = y * y
```

If we want `dz/dx`, we could:

1. **Symbolic differentiation**: Work out the formula by hand → `dz/dx = 8x + 12`
2. **Numerical differentiation**: Use `(f(x+h) - f(x))/h` → Slow and imprecise
3. **Automatic differentiation**: Build a computational graph and apply the chain rule → Fast and exact!

Autodiff gives us **exact derivatives** at **computational cost comparable to the original function**. That's why it's revolutionary.

---

## Part 1: Understanding Computational Graphs

Every computation can be represented as a directed graph.

### Step 1.1: A Simple Example

```cpp
float x = 2.0f;
float y = x * 3.0f;    // y = 3x
float z = y + 1.0f;    // z = y + 1 = 3x + 1
```

This becomes a computational graph:

```
    x (input)
    |
    | *3
    ↓
    y
    |
    | +1
    ↓
    z (output)
```

**Forward pass**: We compute values going forward: `x=2 → y=6 → z=7`

**Backward pass**: We compute gradients going backward using the **chain rule**:

- `dz/dz = 1` (gradient of output w.r.t. itself)
- `dz/dy = 1` (partial derivative of `z = y + 1`)
- `dz/dx = dz/dy * dy/dx = 1 * 3 = 3`

### Step 1.2: The Chain Rule (Your New Best Friend)

The chain rule is the **foundation** of backpropagation:

```
If z = f(y) and y = g(x), then:
dz/dx = (dz/dy) * (dy/dx)
```

For multiple inputs:

```
If z = f(y₁, y₂) where y₁ = g₁(x) and y₂ = g₂(x), then:
dz/dx = (dz/dy₁) * (dy₁/dx) + (dz/dy₂) * (dy₂/dx)
```

**This is all you need to know!** Everything else is just applying this rule systematically.

---

## Part 2: Building the Foundation

Let's build a `Variable` class that wraps a value and tracks its gradient.

### Step 2.1: The Variable Class (Version 1)

Create `autograd.hpp`:

```cpp
#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <iostream>

namespace nn {

class Variable {
private:
    float data_;           // The actual value
    float grad_;           // The gradient (dL/d(this))
    bool requires_grad_;   // Should we track gradients?
    
public:
    Variable(float data, bool requires_grad = false)
        : data_(data)
        , grad_(0.0f)
        , requires_grad_(requires_grad)
    {}
    
    // Getters
    float data() const { return data_; }
    float grad() const { return grad_; }
    bool requires_grad() const { return requires_grad_; }
    
    // Setters
    void set_grad(float grad) { grad_ = grad; }
    void add_grad(float grad) { grad_ += grad; }
    
    // Zero gradients (needed between training steps)
    void zero_grad() { grad_ = 0.0f; }
    
    // Print for debugging
    void print() const {
        std::cout << "Variable(data=" << data_ 
                  << ", grad=" << grad_ << ")\n";
    }
};

} // namespace nn
```

### Step 2.2: Testing the Basic Variable

```cpp
#include "autograd.hpp"

int main() {
    nn::Variable x(5.0f, true);  // requires_grad=true
    nn::Variable y(3.0f, false); // requires_grad=false
    
    x.print();
    y.print();
    
    return 0;
}
```

**Output:**

```
Variable(data=5, grad=0)
Variable(data=3, grad=0)
```

**So far, so boring.** Let's make it actually compute gradients!

---

## Part 3: Adding Operations and Backpropagation

We need to track **how variables are created** so we can backpropagate through them.

### Step 3.1: The Node Concept

Every operation creates a new node in the computational graph. Each node needs to know:

1. Its inputs (parent nodes)
2. How to compute its gradient (backward function)

Let's redesign:

```cpp
#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <iostream>

namespace nn {

class Variable;
using VariablePtr = std::shared_ptr<Variable>;

class Variable : public std::enable_shared_from_this<Variable> {
private:
    float data_;
    float grad_;
    bool requires_grad_;
    
    // Computational graph tracking
    std::vector<VariablePtr> parents_;
    std::function<void()> backward_fn_;
    
public:
    Variable(float data, bool requires_grad = false)
        : data_(data)
        , grad_(0.0f)
        , requires_grad_(requires_grad)
        , backward_fn_(nullptr)
    {}
    
    // Getters
    float data() const { return data_; }
    float grad() const { return grad_; }
    bool requires_grad() const { return requires_grad_; }
    
    void set_data(float data) { data_ = data; }
    void set_grad(float grad) { grad_ = grad; }
    void add_grad(float grad) { grad_ += grad; }
    void zero_grad() { grad_ = 0.0f; }
    
    // Set the backward function
    void set_backward_fn(std::function<void()> fn) {
        backward_fn_ = fn;
    }
    
    // Set parents
    void set_parents(const std::vector<VariablePtr>& parents) {
        parents_ = parents;
    }
    
    // Trigger backpropagation
    void backward(float gradient = 1.0f) {
        // Accumulate gradient
        add_grad(gradient);
        
        // Call the backward function if it exists
        if (backward_fn_) {
            backward_fn_();
        }
    }
    
    void print() const {
        std::cout << "Variable(data=" << data_ 
                  << ", grad=" << grad_ << ")\n";
    }
};

} // namespace nn
```

**Key changes:**

1. `parents_`: Stores the variables that created this one
2. `backward_fn_`: A lambda that knows how to backpropagate through this operation
3. `backward()`: Triggers the backward pass

### Step 3.2: Implementing Addition

Now let's create our first operation: addition!

```cpp
// In nn namespace, after Variable class

VariablePtr operator+(VariablePtr a, VariablePtr b) {
    // Forward pass: compute the result
    auto result = std::make_shared<Variable>(
        a->data() + b->data(),
        a->requires_grad() || b->requires_grad()
    );
    
    // Only set up backprop if gradients are needed
    if (result->requires_grad()) {
        result->set_parents({a, b});
        
        // Define the backward function
        // Captures a, b, and result by value (shared_ptr)
        result->set_backward_fn([a, b, result]() {
            // For addition: dL/da = dL/dresult * 1 = dL/dresult
            //               dL/db = dL/dresult * 1 = dL/dresult
            float grad_output = result->grad();
            
            if (a->requires_grad()) {
                a->add_grad(grad_output);
                // Recursively backprop
                a->backward(0.0f);  // Don't add gradient again
            }
            
            if (b->requires_grad()) {
                b->add_grad(grad_output);
                b->backward(0.0f);
            }
        });
    }
    
    return result;
}
```

**Let's understand the backward function:**

For `z = a + b`:

- `∂z/∂a = 1`
- `∂z/∂b = 1`

By chain rule:

- `dL/da = dL/dz * ∂z/∂a = dL/dz * 1 = dL/dz`
- `dL/db = dL/dz * ∂z/∂b = dL/dz * 1 = dL/dz`

So we just pass the gradient straight through!

### Step 3.3: Testing Addition

```cpp
#include "autograd.hpp"

int main() {
    auto x = std::make_shared<nn::Variable>(2.0f, true);
    auto y = std::make_shared<nn::Variable>(3.0f, true);
    
    auto z = x + y;  // z = 5.0
    
    std::cout << "Forward pass:\n";
    z->print();  // data=5, grad=0
    
    std::cout << "\nBackward pass:\n";
    z->backward(1.0f);  // Start backprop with dL/dz = 1
    
    x->print();  // Should have grad=1
    y->print();  // Should have grad=1
    
    return 0;
}
```

**Expected Output:**

```
Forward pass:
Variable(data=5, grad=0)

Backward pass:
Variable(data=2, grad=1)
Variable(data=3, grad=1)
```

✅ **It works!** We've computed gradients automatically!

### 🎯 Challenge 3.1

Explain why `x` and `y` both have `grad=1`. What would happen if we called `z->backward(2.0f)` instead?

---

## Part 4: More Operations (Multiplication, Powers, etc.)

Let's add more operations to our autograd engine.

### Step 4.1: Multiplication

For `z = a * b`:

- `∂z/∂a = b`
- `∂z/∂b = a`

```cpp
VariablePtr operator*(VariablePtr a, VariablePtr b) {
    auto result = std::make_shared<Variable>(
        a->data() * b->data(),
        a->requires_grad() || b->requires_grad()
    );
    
    if (result->requires_grad()) {
        result->set_parents({a, b});
        
        result->set_backward_fn([a, b, result]() {
            float grad_output = result->grad();
            
            if (a->requires_grad()) {
                // dL/da = dL/dz * ∂z/∂a = dL/dz * b
                a->add_grad(grad_output * b->data());
                a->backward(0.0f);
            }
            
            if (b->requires_grad()) {
                // dL/db = dL/dz * ∂z/∂b = dL/dz * a
                b->add_grad(grad_output * a->data());
                b->backward(0.0f);
            }
        });
    }
    
    return result;
}
```

### Step 4.2: Scalar Multiplication

Often we want to multiply by a constant:

```cpp
VariablePtr operator*(VariablePtr a, float scalar) {
    auto result = std::make_shared<Variable>(
        a->data() * scalar,
        a->requires_grad()
    );
    
    if (result->requires_grad()) {
        result->set_parents({a});
        
        result->set_backward_fn([a, scalar, result]() {
            // dL/da = dL/dz * scalar
            if (a->requires_grad()) {
                a->add_grad(result->grad() * scalar);
                a->backward(0.0f);
            }
        });
    }
    
    return result;
}

// For scalar * Variable
VariablePtr operator*(float scalar, VariablePtr a) {
    return a * scalar;
}
```

### Step 4.3: Power Function

For `z = aⁿ`:

- `∂z/∂a = n * a^(n-1)`

```cpp
VariablePtr pow(VariablePtr a, float n) {
    auto result = std::make_shared<Variable>(
        std::pow(a->data(), n),
        a->requires_grad()
    );
    
    if (result->requires_grad()) {
        result->set_parents({a});
        
        result->set_backward_fn([a, n, result]() {
            if (a->requires_grad()) {
                // dL/da = dL/dz * n * a^(n-1)
                float grad = result->grad() * n * std::pow(a->data(), n - 1);
                a->add_grad(grad);
                a->backward(0.0f);
            }
        });
    }
    
    return result;
}
```

### Step 4.4: Testing Complex Expressions

Now we can compute gradients of complex expressions!

```cpp
#include "autograd.hpp"
#include <cmath>

int main() {
    // Let's compute the gradient of: f(x) = (x + 2) * (x + 2)
    // Mathematically: f(x) = x² + 4x + 4
    // df/dx = 2x + 4
    // At x=3: df/dx = 2*3 + 4 = 10
    
    auto x = std::make_shared<nn::Variable>(3.0f, true);
    auto two = std::make_shared<nn::Variable>(2.0f, false);
    
    auto a = x + two;      // a = x + 2 = 5
    auto b = x + two;      // b = x + 2 = 5
    auto f = a * b;        // f = (x+2)*(x+2) = 25
    
    std::cout << "f(3) = " << f->data() << "\n";
    
    f->backward(1.0f);
    
    std::cout << "df/dx at x=3: " << x->grad() << "\n";
    std::cout << "Expected: 10\n";
    
    return 0;
}
```

**Expected output:**

```
f(3) = 25
df/dx at x=3: 10
Expected: 10
```

### 🎯 Challenge 4.1

Implement these operations:

1. Subtraction: `z = a - b`
2. Division: `z = a / b`
3. Negation: `z = -a`

**Hint:** For division, remember that `∂(a/b)/∂b = -a/b²`

---

## Part 5: The Problem with Our Current Approach

Let's try something:

```cpp
auto x = std::make_shared<nn::Variable>(2.0f, true);
auto y = x + x;  // y = 2x
y->backward(1.0f);

std::cout << "x.grad = " << x->grad() << "\n";
// Output: x.grad = 2 ✓ Correct!
```

This works! But now try:

```cpp
auto x = std::make_shared<nn::Variable>(2.0f, true);

auto y = x + x;  // y = 2x
y->backward(1.0f);
std::cout << "After 1st backward: x.grad = " << x->grad() << "\n";

auto z = x + x;  // z = 2x
z->backward(1.0f);
std::cout << "After 2nd backward: x.grad = " << x->grad() << "\n";
```

**Output:**

```
After 1st backward: x.grad = 2
After 2nd backward: x.grad = 4  ← Wrong! Should be 2
```

**The problem?** We're **accumulating** gradients, but we never reset them!

### Step 5.1: Gradient Zeroing

In PyTorch, you call `optimizer.zero_grad()` before each backward pass. Let's implement this:

```cpp
// Add to Variable class
void zero_grad() { 
    grad_ = 0.0f; 
}
```

Now use it correctly:

```cpp
auto x = std::make_shared<nn::Variable>(2.0f, true);

auto y = x + x;
y->backward(1.0f);
std::cout << "After 1st: x.grad = " << x->grad() << "\n";

x->zero_grad();  // ← Important!

auto z = x + x;
z->backward(1.0f);
std::cout << "After 2nd: x.grad = " << x->grad() << "\n";
```

**Output:**

```
After 1st: x.grad = 2
After 2nd: x.grad = 2  ✓ Correct!
```

---

## Part 6: Topological Sort and Backpropagation Order

There's another subtle issue with our current implementation.

### Step 6.1: The Problem

Consider:

```cpp
a = x + y   // a depends on x, y
b = a * 2   // b depends on a
c = a + 3   // c depends on a
d = b + c   // d depends on b, c
```

When we call `d->backward()`, we need to compute gradients in the **reverse topological order**:

```
d → b → a → x, y
  → c → a → x, y
```

Notice `a` is visited **twice** (from both `b` and `c`). We need to accumulate its gradient from **both paths** before propagating to `x` and `y`.

Our current recursive approach might not handle this correctly!

### Step 6.2: The Correct Approach (Topological Sort)

```cpp
class Variable {
    // ... previous code ...
    
    void backward(float gradient = 1.0f) {
        // Build topological order
        std::vector<VariablePtr> topo_order;
        std::unordered_set<Variable*> visited;
        
        std::function<void(VariablePtr)> build_topo = [&](VariablePtr v) {
            if (visited.count(v.get()) > 0) return;
            visited.insert(v.get());
            
            for (auto& parent : v->parents_) {
                build_topo(parent);
            }
            
            topo_order.push_back(v);
        };
        
        build_topo(shared_from_this());
        
        // Initialize gradient
        this->set_grad(gradient);
        
        // Backpropagate in reverse topological order
        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            auto v = *it;
            if (v->backward_fn_) {
                v->backward_fn_();
            }
        }
    }
};
```

But wait - this changes our backward function signature! Let's update our operations:

```cpp
VariablePtr operator+(VariablePtr a, VariablePtr b) {
    auto result = std::make_shared<Variable>(
        a->data() + b->data(),
        a->requires_grad() || b->requires_grad()
    );
    
    if (result->requires_grad()) {
        result->set_parents({a, b});
        
        result->set_backward_fn([a, b, result]() {
            float grad_output = result->grad();
            
            if (a->requires_grad()) {
                a->add_grad(grad_output);  // Just accumulate, no recursive call
            }
            
            if (b->requires_grad()) {
                b->add_grad(grad_output);
            }
        });
    }
    
    return result;
}
```

Now the topological sort handles the traversal order!

### 🎯 Challenge 6.1

Draw the computational graph for this expression:

```cpp
f = (x + y) * (x + 2)
```

Then trace the backward pass manually, showing what gradients are computed at each step.

---

## Part 7: Activation Functions

Neural networks need **non-linear** activation functions. Let's implement the most common ones.

### Step 7.1: Sigmoid

```cpp
VariablePtr sigmoid(VariablePtr x) {
    // σ(x) = 1 / (1 + e^(-x))
    float sig_x = 1.0f / (1.0f + std::exp(-x->data()));
    
    auto result = std::make_shared<Variable>(sig_x, x->requires_grad());
    
    if (result->requires_grad()) {
        result->set_parents({x});
        
        result->set_backward_fn([x, result, sig_x]() {
            // d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
            float grad = result->grad() * sig_x * (1.0f - sig_x);
            if (x->requires_grad()) {
                x->add_grad(grad);
            }
        });
    }
    
    return result;
}
```

**Why this derivative?**

Sigmoidhas a beautiful property:

```
If σ(x) = 1/(1 + e^(-x)), then:
dσ/dx = σ(x) * (1 - σ(x))
```

This makes backprop very efficient!

### Step 7.2: ReLU (Rectified Linear Unit)

```cpp
VariablePtr relu(VariablePtr x) {
    // ReLU(x) = max(0, x)
    float relu_x = std::max(0.0f, x->data());
    
    auto result = std::make_shared<Variable>(relu_x, x->requires_grad());
    
    if (result->requires_grad()) {
        result->set_parents({x});
        
        result->set_backward_fn([x, result]() {
            // d(ReLU)/dx = 1 if x > 0, else 0
            float grad = result->grad() * (x->data() > 0.0f ? 1.0f : 0.0f);
            if (x->requires_grad()) {
                x->add_grad(grad);
            }
        });
    }
    
    return result;
}
```

**ReLU is the most popular activation** because:

1. Simple and fast
2. Doesn't saturate (unlike sigmoid/tanh)
3. Works well in practice

### Step 7.3: Tanh

```cpp
VariablePtr tanh(VariablePtr x) {
    float tanh_x = std::tanh(x->data());
    
    auto result = std::make_shared<Variable>(tanh_x, x->requires_grad());
    
    if (result->requires_grad()) {
        result->set_parents({x});
        
        result->set_backward_fn([x, result, tanh_x]() {
            // d(tanh)/dx = 1 - tanh²(x)
            float grad = result->grad() * (1.0f - tanh_x * tanh_x);
            if (x->requires_grad()) {
                x->add_grad(grad);
            }
        });
    }
    
    return result;
}
```

---

## Part 8: Building a Simple Neural Network

Let's put it all together and build a **real neural network** that learns!

### Step 8.1: The XOR Problem

XOR is a classic problem that a single perceptron can't solve, but a 2-layer network can.

```
Input    Output
0  0  →  0
0  1  →  1
1  0  →  1
1  1  →  0
```

### Step 8.2: Neural Network Implementation

```cpp
#include "autograd.hpp"
#include <iostream>
#include <vector>
#include <random>

class NeuralNetwork {
private:
    // Layer 1: 2 inputs → 4 hidden
    std::vector<std::vector<nn::VariablePtr>> W1;  // 2x4
    std::vector<nn::VariablePtr> b1;               // 4
    
    // Layer 2: 4 hidden → 1 output
    std::vector<std::vector<nn::VariablePtr>> W2;  // 4x1
    std::vector<nn::VariablePtr> b2;               // 1
    
    std::mt19937 rng;
    std::normal_distribution<float> dist;
    
public:
    NeuralNetwork() : rng(42), dist(0.0f, 0.5f) {
        // Initialize weights randomly
        
        // W1: 2x4
        W1.resize(2);
        for (int i = 0; i < 2; ++i) {
            W1[i].resize(4);
            for (int j = 0; j < 4; ++j) {
                W1[i][j] = std::make_shared<nn::Variable>(dist(rng), true);
            }
        }
        
        // b1: 4
        b1.resize(4);
        for (int i = 0; i < 4; ++i) {
            b1[i] = std::make_shared<nn::Variable>(dist(rng), true);
        }
        
        // W2: 4x1
        W2.resize(4);
        for (int i = 0; i < 4; ++i) {
            W2[i].resize(1);
            W2[i][0] = std::make_shared<nn::Variable>(dist(rng), true);
        }
        
        // b2: 1
        b2.resize(1);
        b2[0] = std::make_shared<nn::Variable>(dist(rng), true);
    }
    
    nn::VariablePtr forward(float x1_val, float x2_val) {
        auto x1 = std::make_shared<nn::Variable>(x1_val, false);
        auto x2 = std::make_shared<nn::Variable>(x2_val, false);
        
        // Layer 1: hidden = sigmoid(x @ W1 + b1)
        std::vector<nn::VariablePtr> hidden(4);
        for (int j = 0; j < 4; ++j) {
            auto z = x1 * W1[0][j] + x2 * W1[1][j] + b1[j];
            hidden[j] = nn::sigmoid(z);
        }
        
        // Layer 2: output = sigmoid(hidden @ W2 + b2)
        auto z_out = b2[0];
        for (int i = 0; i < 4; ++i) {
            z_out = z_out + hidden[i] * W2[i][0];
        }
        auto output = nn::sigmoid(z_out);
        
        return output;
    }
    
    void zero_grad() {
        for (auto& row : W1) {
            for (auto& w : row) w->zero_grad();
        }
        for (auto& b : b1) b->zero_grad();
        
        for (auto& row : W2) {
            for (auto& w : row) w->zero_grad();
        }
        for (auto& b : b2) b->zero_grad();
    }
    
    void update_weights(float lr) {
        // Gradient descent: w = w - lr * dL/dw
        for (auto& row : W1) {
            for (auto& w : row) {
                w->set_data(w->data() - lr * w->grad());
            }
        }
        for (auto& b : b1) {
            b->set_data(b->data() - lr * b->grad());
        }
        
        for (auto& row : W2) {
            for (auto& w : row) {
                w->set_data(w->data() - lr * w->grad());
            }
        }
        for (auto& b : b2) {
            b->set_data(b->data() - lr * b->grad());
        }
    }
};

int main() {
    NeuralNetwork nn;
    
    // XOR dataset
    std::vector<std::pair<std::pair<float, float>, float>> dataset = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 0}
    };
    
    float learning_rate = 0.5f;
    int epochs = 5000;
    
    std::cout << "Training...\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        
        for (auto& [input, target] : dataset) {
            nn.zero_grad();
            
            // Forward pass
            auto output = nn.forward(input.first, input.second);
            
            // Compute loss: MSE = (output - target)²
            auto diff = output + std::make_shared<nn::Variable>(-target, false);
            auto loss = nn::pow(diff, 2.0f);
            
            total_loss += loss->data();
            
            // Backward pass
            loss->backward();
            
            // Update weights
            nn.update_weights(learning_rate);
        }
        
        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / 4.0f << "\n";
        }
    }
    
    std::cout << "\nTesting:\n";
    for (auto& [input, target] : dataset) {
        auto output = nn.forward(input.first, input.second);
        std::cout << input.first << " XOR " << input.second 
                  << " = " << output->data() 
                  << " (target: " << target << ")\n";
    }
    
    return 0;
}
```

**Run this and you'll see the network learn XOR from scratch!**

### 🎯 Challenge 8.1

The network above sometimes fails to converge. Why? Research:

1. Weight initialization strategies (Xavier, He initialization)
2. Learning rate schedules
3. Different loss functions

Try implementing these improvements!

---

## Part 9: Comparison with PyTorch

Let's see how our autograd compares to PyTorch:

### Our Implementation

```cpp
auto x = std::make_shared<nn::Variable>(3.0f, true);
auto y = x * x + x * 2.0f + 1.0f;
y->backward();
std::cout << x->grad() << "\n";  // 8.0
```

### PyTorch Equivalent

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x * x + x * 2.0 + 1.0
y.backward()
print(x.grad)  # tensor(8.0)
```

**We've built the same thing!** 🎉

---

## Part 10: Optimization Opportunities

Our implementation is educational but slow. Here are optimizations used in production frameworks:

### 10.1: Operator Fusion

Instead of:

```cpp
a = x + 1
b = a * 2
c = b + 3
```

Fuse into:

```cpp
c = (x + 1) * 2 + 3
```

Fewer intermediate allocations, better cache usage.

### 10.2: In-Place Operations

PyTorch has `add_()` (in-place) vs `add()` (out-of-place):

```python
x.add_(y)  # Modifies x directly, no new allocation
```

### 10.3: Static Graphs (TensorFlow 1.x style)

Build the graph once, reuse many times:

- Analyze the graph for optimizations
- Allocate memory once
- JIT compile

### 10.4: Lazy Evaluation

Don't compute until absolutely necessary. XLA (Accelerated Linear Algebra) does this.

---

## Summary and What's Next

### What You've Built

🎉 **Congratulations!** You now have:

- ✅ A working automatic differentiation engine
- ✅ Support for arbitrary computational graphs
- ✅ Common activation functions
- ✅ A neural network that learns XOR from scratch

### Key Insights

1. **Computational graphs** represent calculations as DAGs
2. **Forward pass** computes values; **backward pass** computes gradients
3. **Chain rule** is applied automatically through the graph
4. **Topological sort** ensures correct gradient flow
5. **Gradient accumulation** requires careful `zero_grad()` calls

### What's Missing

- Broadcasting support
- Batch processing (currently only single samples)
- GPU support
- Higher-order derivatives
- Dynamic vs static graphs
- Memory optimization

---

## 🎯 Final Challenge: Implement Mean Squared Error

Create an MSE loss function:

```cpp
VariablePtr mse_loss(VariablePtr pred, VariablePtr target) {
    // MSE = (pred - target)²
    // Your code here
}
```

Requirements:

1. Compute the squared difference
2. Support backpropagation
3. Test with a simple example

---

## Further Reading

1. **Automatic Differentiation**:
   - "Automatic Differentiation in Machine Learning: a Survey" (Baydin et al.)
   - JAX documentation on autodiff

2. **Computational Graphs**:
   - PyTorch autograd internals
   - TensorFlow graph execution

3. **Optimization**:
   - XLA (Accelerated Linear Algebra) compiler
   - TorchScript

4. **Theory**:
   - Chain rule and multivariate calculus
   - Topological sorting algorithms

---

**Next Tutorial**: Building Training Infrastructure (Optimizers, Loss Functions, Data Loaders)

Now let's make training neural networks convenient and efficient! 🚀
