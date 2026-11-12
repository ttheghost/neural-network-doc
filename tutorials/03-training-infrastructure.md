# Tutorial 03: Building Training Infrastructure

> **Learning Goal**: By the end of this tutorial, you'll have built the complete training infrastructure: optimizers (SGD, Adam), loss functions (MSE, Cross-Entropy), and data loaders. This is what transforms your neural network from a mathematical curiosity into a practical machine learning system.

## The Training Loop: What Are We Building?

Every machine learning framework has the same training pattern:

```python
# PyTorch-style training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch.inputs)
        loss = loss_function(predictions, batch.targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
```

We're going to build **all** of these components in C++!

---

## Part 1: Loss Functions (How Wrong Are We?)

Loss functions measure how far our predictions are from the truth. They're the **objective** we're trying to minimize.

### Step 1.1: Understanding Loss Functions

**Two main categories:**

1. **Regression** (predicting continuous values)
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Huber Loss

2. **Classification** (predicting categories)
   - Binary Cross-Entropy
   - Categorical Cross-Entropy
   - Hinge Loss (for SVMs)

### Step 1.2: Mean Squared Error (MSE)

**Formula**: `MSE = (1/n) * Σ(yᵢ - ŷᵢ)²`

This is perfect for regression tasks.

Create `loss.hpp`:

```cpp
#pragma once
#include "autograd.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace nn {

class Loss {
public:
    virtual ~Loss() = default;
    
    // Compute loss for a single sample
    virtual VariablePtr operator()(VariablePtr pred, VariablePtr target) = 0;
    
    // Compute loss for a batch
    virtual VariablePtr batch_loss(
        const std::vector<VariablePtr>& preds,
        const std::vector<VariablePtr>& targets
    ) = 0;
};

class MSELoss : public Loss {
public:
    VariablePtr operator()(VariablePtr pred, VariablePtr target) override {
        // MSE = (pred - target)²
        auto diff = pred + (target * -1.0f);
        return pow(diff, 2.0f);
    }
    
    VariablePtr batch_loss(
        const std::vector<VariablePtr>& preds,
        const std::vector<VariablePtr>& targets
    ) override {
        if (preds.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have same size");
        }
        
        // Average loss across batch
        auto total_loss = std::make_shared<Variable>(0.0f, true);
        
        for (size_t i = 0; i < preds.size(); ++i) {
            auto loss_i = (*this)(preds[i], targets[i]);
            total_loss = total_loss + loss_i;
        }
        
        // Divide by batch size
        return total_loss * (1.0f / static_cast<float>(preds.size()));
    }
};

} // namespace nn
```

**Why average?** Averaging makes the loss independent of batch size, which helps with:

- Consistent learning rates across different batch sizes
- Comparable losses between training runs

### Step 1.3: Binary Cross-Entropy Loss

For binary classification (0 or 1):

**Formula**: `BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]`

```cpp
class BCELoss : public Loss {
public:
    VariablePtr operator()(VariablePtr pred, VariablePtr target) override {
        // BCE = -[t*log(p) + (1-t)*log(1-p)]
        // For numerical stability, we clip predictions
        float eps = 1e-7f;
        float p = std::max(eps, std::min(1.0f - eps, pred->data()));
        float t = target->data();
        
        float loss_val = -(t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
        
        auto result = std::make_shared<Variable>(loss_val, pred->requires_grad());
        
        if (result->requires_grad()) {
            result->set_parents({pred});
            
            result->set_backward_fn([pred, target, result, p, t]() {
                // Derivative: dL/dp = -(t/p - (1-t)/(1-p))
                //                   = (p - t) / (p * (1 - p))
                float grad = (p - t) / (p * (1.0f - p));
                
                if (pred->requires_grad()) {
                    pred->add_grad(result->grad() * grad);
                }
            });
        }
        
        return result;
    }
    
    VariablePtr batch_loss(
        const std::vector<VariablePtr>& preds,
        const std::vector<VariablePtr>& targets
    ) override {
        auto total_loss = std::make_shared<Variable>(0.0f, true);
        
        for (size_t i = 0; i < preds.size(); ++i) {
            total_loss = total_loss + (*this)(preds[i], targets[i]);
        }
        
        return total_loss * (1.0f / static_cast<float>(preds.size()));
    }
};
```

**Note on numerical stability**: We clip predictions to `[ε, 1-ε]` to avoid `log(0) = -∞`.

### Step 1.4: Categorical Cross-Entropy Loss

For multi-class classification:

**Formula**: `CCE = -Σ(yᵢ * log(ŷᵢ))`

where `y` is one-hot encoded.

```cpp
class CrossEntropyLoss : public Loss {
public:
    // For multi-class: pred and target are vectors
    VariablePtr operator()(
        const std::vector<VariablePtr>& pred_probs,
        const std::vector<VariablePtr>& target_one_hot
    ) {
        if (pred_probs.size() != target_one_hot.size()) {
            throw std::invalid_argument("Size mismatch");
        }
        
        float eps = 1e-7f;
        float loss_val = 0.0f;
        
        for (size_t i = 0; i < pred_probs.size(); ++i) {
            float p = std::max(eps, pred_probs[i]->data());
            float t = target_one_hot[i]->data();
            loss_val -= t * std::log(p);
        }
        
        // ... (gradient computation similar to BCE)
        
        return std::make_shared<Variable>(loss_val, true);
    }
    
    // For single-target version (class index)
    VariablePtr operator()(VariablePtr pred, VariablePtr target) override {
        // Simplified: just return the negative log probability of true class
        float eps = 1e-7f;
        float p = std::max(eps, pred->data());
        float loss_val = -std::log(p);
        
        return std::make_shared<Variable>(loss_val, pred->requires_grad());
    }
    
    VariablePtr batch_loss(
        const std::vector<VariablePtr>& preds,
        const std::vector<VariablePtr>& targets
    ) override {
        auto total_loss = std::make_shared<Variable>(0.0f, true);
        
        for (size_t i = 0; i < preds.size(); ++i) {
            total_loss = total_loss + (*this)(preds[i], targets[i]);
        }
        
        return total_loss * (1.0f / static_cast<float>(preds.size()));
    }
};
```

### 🎯 Challenge 1.1

Implement **Mean Absolute Error (MAE)**: `MAE = (1/n) * Σ|yᵢ - ŷᵢ|`

**Hint**: You'll need to implement an `abs()` function with proper gradient handling. Remember: `d|x|/dx = sign(x)`.

---

## Part 2: Optimizers (How Do We Update Weights?)

Optimizers take gradients and update parameters. Different optimizers use different strategies.

### Step 2.1: The Optimizer Interface

Create `optimizer.hpp`:

```cpp
#pragma once
#include "autograd.hpp"
#include <vector>
#include <memory>

namespace nn {

class Optimizer {
protected:
    std::vector<VariablePtr> parameters_;
    float learning_rate_;
    
public:
    Optimizer(const std::vector<VariablePtr>& params, float lr)
        : parameters_(params), learning_rate_(lr) {}
    
    virtual ~Optimizer() = default;
    
    // Update all parameters
    virtual void step() = 0;
    
    // Zero all gradients
    virtual void zero_grad() {
        for (auto& param : parameters_) {
            param->zero_grad();
        }
    }
    
    // Change learning rate
    void set_learning_rate(float lr) {
        learning_rate_ = lr;
    }
    
    float get_learning_rate() const {
        return learning_rate_;
    }
};

} // namespace nn
```

### Step 2.2: Stochastic Gradient Descent (SGD)

The simplest optimizer:

**Update rule**: `θ = θ - η * ∇L`

where:

- `θ` = parameters
- `η` = learning rate
- `∇L` = gradient

```cpp
class SGD : public Optimizer {
public:
    SGD(const std::vector<VariablePtr>& params, float lr)
        : Optimizer(params, lr) {}
    
    void step() override {
        for (auto& param : parameters_) {
            if (!param->requires_grad()) continue;
            
            // θ = θ - η * ∇θ
            float new_data = param->data() - learning_rate_ * param->grad();
            param->set_data(new_data);
        }
    }
};
```

**Simple but effective!** Most research still uses SGD variants.

### Step 2.3: SGD with Momentum

**Problem with vanilla SGD**: It oscillates in ravines (narrow valleys).

**Solution**: Add momentum - remember previous update direction.

**Update rule**:

```
v = β * v + ∇L
θ = θ - η * v
```

where `β` is typically 0.9.

```cpp
class SGDMomentum : public Optimizer {
private:
    float momentum_;
    std::vector<float> velocity_;
    
public:
    SGDMomentum(const std::vector<VariablePtr>& params, float lr, float momentum = 0.9f)
        : Optimizer(params, lr)
        , momentum_(momentum)
        , velocity_(params.size(), 0.0f) {}
    
    void step() override {
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            if (!param->requires_grad()) continue;
            
            // v = β*v + ∇θ
            velocity_[i] = momentum_ * velocity_[i] + param->grad();
            
            // θ = θ - η*v
            float new_data = param->data() - learning_rate_ * velocity_[i];
            param->set_data(new_data);
        }
    }
};
```

**Visualization**:

- Without momentum: Ball rolling down a hill, easily gets stuck
- With momentum: Ball with inertia, can roll over small bumps

### Step 2.4: Adam (Adaptive Moment Estimation)

**The most popular optimizer!** Combines momentum with adaptive learning rates.

**Update rules**:

```
m = β₁ * m + (1 - β₁) * ∇L        (first moment - momentum)
v = β₂ * v + (1 - β₂) * ∇L²       (second moment - squared gradients)

m̂ = m / (1 - β₁ᵗ)                 (bias correction)
v̂ = v / (1 - β₂ᵗ)

θ = θ - η * m̂ / (√v̂ + ε)
```

Typical values: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`

```cpp
class Adam : public Optimizer {
private:
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_;  // timestep
    
    std::vector<float> m_;  // first moment
    std::vector<float> v_;  // second moment
    
public:
    Adam(const std::vector<VariablePtr>& params, 
         float lr,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float epsilon = 1e-8f)
        : Optimizer(params, lr)
        , beta1_(beta1)
        , beta2_(beta2)
        , epsilon_(epsilon)
        , t_(0)
        , m_(params.size(), 0.0f)
        , v_(params.size(), 0.0f)
    {}
    
    void step() override {
        t_++;  // Increment timestep
        
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            if (!param->requires_grad()) continue;
            
            float grad = param->grad();
            
            // Update biased first moment estimate
            m_[i] = beta1_ * m_[i] + (1.0f - beta1_) * grad;
            
            // Update biased second raw moment estimate
            v_[i] = beta2_ * v_[i] + (1.0f - beta2_) * grad * grad;
            
            // Compute bias-corrected first moment estimate
            float m_hat = m_[i] / (1.0f - std::pow(beta1_, t_));
            
            // Compute bias-corrected second raw moment estimate
            float v_hat = v_[i] / (1.0f - std::pow(beta2_, t_));
            
            // Update parameters
            float new_data = param->data() - learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            param->set_data(new_data);
        }
    }
};
```

**Why Adam is popular**:

1. **Adaptive learning rates**: Each parameter gets its own effective learning rate
2. **Momentum**: Smooths out updates
3. **Bias correction**: Works well from the start (t=0)
4. **Robust**: Works well with default hyperparameters

### Step 2.5: Quick Comparison

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| **SGD** | Simple, well-understood | Needs tuning, slow | Convex problems |
| **SGD+Momentum** | Faster convergence | Still needs tuning | Deep networks |
| **Adam** | Adaptive, robust | Can generalize worse | Most tasks |

### 🎯 Challenge 2.1

Implement **RMSprop**, which is like Adam but without the momentum term:

```
v = β * v + (1 - β) * ∇L²
θ = θ - η * ∇L / (√v + ε)
```

Typical: `β = 0.99`, `ε = 1e-8`

---

## Part 3: Data Loading (Feeding the Beast)

Neural networks need data - lots of it, efficiently organized.

### Step 3.1: Dataset Interface

Create `dataset.hpp`:

```cpp
#pragma once
#include <vector>
#include <utility>
#include <stdexcept>

namespace nn {

template<typename InputType, typename TargetType>
class Dataset {
public:
    virtual ~Dataset() = default;
    
    // Get a single sample
    virtual std::pair<InputType, TargetType> get(size_t index) const = 0;
    
    // Get dataset size
    virtual size_t size() const = 0;
    
    // Shuffle the dataset (optional)
    virtual void shuffle() {}
};

} // namespace nn
```

### Step 3.2: Simple In-Memory Dataset

```cpp
template<typename InputType, typename TargetType>
class MemoryDataset : public Dataset<InputType, TargetType> {
private:
    std::vector<std::pair<InputType, TargetType>> data_;
    
public:
    MemoryDataset(const std::vector<std::pair<InputType, TargetType>>& data)
        : data_(data) {}
    
    std::pair<InputType, TargetType> get(size_t index) const override {
        if (index >= data_.size()) {
            throw std::out_of_range("Index out of range");
        }
        return data_[index];
    }
    
    size_t size() const override {
        return data_.size();
    }
    
    void shuffle() override {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data_.begin(), data_.end(), g);
    }
    
    // Add sample
    void add(const InputType& input, const TargetType& target) {
        data_.emplace_back(input, target);
    }
};
```

### Step 3.3: DataLoader (Batching)

A DataLoader takes a Dataset and yields batches:

```cpp
template<typename InputType, typename TargetType>
class DataLoader {
private:
    Dataset<InputType, TargetType>* dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
    size_t current_idx_;
    
public:
    DataLoader(Dataset<InputType, TargetType>* dataset, 
               size_t batch_size,
               bool shuffle = true)
        : dataset_(dataset)
        , batch_size_(batch_size)
        , shuffle_(shuffle)
        , current_idx_(0)
    {
        reset();
    }
    
    void reset() {
        // Initialize indices
        indices_.resize(dataset_->size());
        for (size_t i = 0; i < indices_.size(); ++i) {
            indices_[i] = i;
        }
        
        // Shuffle if requested
        if (shuffle_) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
        
        current_idx_ = 0;
    }
    
    bool has_next() const {
        return current_idx_ < indices_.size();
    }
    
    std::vector<std::pair<InputType, TargetType>> next_batch() {
        if (!has_next()) {
            throw std::runtime_error("No more batches");
        }
        
        std::vector<std::pair<InputType, TargetType>> batch;
        
        size_t batch_end = std::min(current_idx_ + batch_size_, indices_.size());
        
        for (size_t i = current_idx_; i < batch_end; ++i) {
            batch.push_back(dataset_->get(indices_[i]));
        }
        
        current_idx_ = batch_end;
        
        return batch;
    }
    
    size_t num_batches() const {
        return (dataset_->size() + batch_size_ - 1) / batch_size_;
    }
};
```

**Usage example**:

```cpp
// Create dataset
MemoryDataset<std::vector<float>, float> dataset;
dataset.add({1.0, 2.0}, 3.0);
dataset.add({2.0, 3.0}, 5.0);
// ... more data

// Create dataloader
DataLoader loader(&dataset, batch_size=32, shuffle=true);

// Iterate
while (loader.has_next()) {
    auto batch = loader.next_batch();
    // Train on batch
}
```

### 🎯 Challenge 3.1

Implement a **CSV Dataset** that reads data from a CSV file line by line, without loading everything into memory. This is important for large datasets!

**Hint**: You'll need to keep a file handle and use `seekg()` to jump to specific lines.

---

## Part 4: Putting It All Together - Complete Training Loop

Now let's combine everything into a beautiful, PyTorch-style training loop!

### Step 4.1: The Training Framework

```cpp
#pragma once
#include "autograd.hpp"
#include "optimizer.hpp"
#include "loss.hpp"
#include "dataset.hpp"
#include <iostream>
#include <iomanip>

namespace nn {

template<typename Model>
class Trainer {
private:
    Model& model_;
    Optimizer& optimizer_;
    Loss& loss_fn_;
    
public:
    Trainer(Model& model, Optimizer& optimizer, Loss& loss_fn)
        : model_(model)
        , optimizer_(optimizer)
        , loss_fn_(loss_fn)
    {}
    
    void train_epoch(DataLoader<std::vector<float>, float>& loader) {
        loader.reset();
        
        float epoch_loss = 0.0f;
        size_t num_batches = 0;
        
        while (loader.has_next()) {
            auto batch = loader.next_batch();
            
            // Zero gradients
            optimizer_.zero_grad();
            
            // Forward pass
            std::vector<VariablePtr> predictions;
            std::vector<VariablePtr> targets;
            
            for (const auto& [input, target] : batch) {
                auto pred = model_.forward(input);
                predictions.push_back(pred);
                targets.push_back(std::make_shared<Variable>(target, false));
            }
            
            // Compute loss
            auto loss = loss_fn_.batch_loss(predictions, targets);
            epoch_loss += loss->data();
            
            // Backward pass
            loss->backward();
            
            // Update weights
            optimizer_.step();
            
            num_batches++;
        }
        
        std::cout << "Average Loss: " << (epoch_loss / num_batches) << "\n";
    }
    
    void train(DataLoader<std::vector<float>, float>& train_loader,
               int num_epochs,
               DataLoader<std::vector<float>, float>* val_loader = nullptr) {
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << "\n";
            
            // Training
            train_epoch(train_loader);
            
            // Validation (if provided)
            if (val_loader) {
                validate(*val_loader);
            }
            
            std::cout << "---\n";
        }
    }
    
    void validate(DataLoader<std::vector<float>, float>& loader) {
        loader.reset();
        
        float val_loss = 0.0f;
        size_t num_batches = 0;
        
        while (loader.has_next()) {
            auto batch = loader.next_batch();
            
            std::vector<VariablePtr> predictions;
            std::vector<VariablePtr> targets;
            
            for (const auto& [input, target] : batch) {
                auto pred = model_.forward(input);
                predictions.push_back(pred);
                targets.push_back(std::make_shared<Variable>(target, false));
            }
            
            auto loss = loss_fn_.batch_loss(predictions, targets);
            val_loss += loss->data();
            
            num_batches++;
        }
        
        std::cout << "Validation Loss: " << (val_loss / num_batches) << "\n";
    }
};

} // namespace nn
```

### Step 4.2: Complete Example - Regression on Synthetic Data

```cpp
#include "autograd.hpp"
#include "optimizer.hpp"
#include "loss.hpp"
#include "dataset.hpp"
#include "trainer.hpp"
#include <random>
#include <iostream>

// Simple linear regression model: y = w*x + b
class LinearModel {
private:
    nn::VariablePtr w_;
    nn::VariablePtr b_;
    
public:
    LinearModel() {
        // Initialize with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        w_ = std::make_shared<nn::Variable>(dist(gen), true);
        b_ = std::make_shared<nn::Variable>(dist(gen), true);
    }
    
    nn::VariablePtr forward(const std::vector<float>& input) {
        // For simplicity, assuming single input
        auto x = std::make_shared<nn::Variable>(input[0], false);
        return x * w_ + b_;
    }
    
    std::vector<nn::VariablePtr> parameters() {
        return {w_, b_};
    }
    
    void print_params() {
        std::cout << "w = " << w_->data() << ", b = " << b_->data() << "\n";
    }
};

int main() {
    // Generate synthetic data: y = 3x + 2 + noise
    nn::MemoryDataset<std::vector<float>, float> dataset;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    for (int i = 0; i < 1000; ++i) {
        float x = static_cast<float>(i) / 100.0f;  // 0 to 10
        float y = 3.0f * x + 2.0f + noise(gen);
        dataset.add({x}, y);
    }
    
    // Split into train/val
    nn::MemoryDataset<std::vector<float>, float> train_dataset;
    nn::MemoryDataset<std::vector<float>, float> val_dataset;
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto sample = dataset.get(i);
        if (i < 800) {
            train_dataset.add(sample.first, sample.second);
        } else {
            val_dataset.add(sample.first, sample.second);
        }
    }
    
    // Create dataloaders
    nn::DataLoader train_loader(&train_dataset, 32, true);
    nn::DataLoader val_loader(&val_dataset, 32, false);
    
    // Create model
    LinearModel model;
    std::cout << "Initial parameters:\n";
    model.print_params();
    
    // Create optimizer and loss
    nn::Adam optimizer(model.parameters(), 0.1f);
    nn::MSELoss loss_fn;
    
    // Create trainer
    nn::Trainer trainer(model, optimizer, loss_fn);
    
    // Train!
    std::cout << "\nTraining...\n";
    trainer.train(train_loader, 20, &val_loader);
    
    std::cout << "\nFinal parameters:\n";
    model.print_params();
    std::cout << "Expected: w ≈ 3.0, b ≈ 2.0\n";
    
    return 0;
}
```

**You should see the model learn `w ≈ 3.0` and `b ≈ 2.0`!**

### 🎯 Challenge 4.1

Add these features to the Trainer:

1. **Early stopping**: Stop training if validation loss doesn't improve for N epochs
2. **Learning rate scheduling**: Reduce learning rate by factor when validation plateaus
3. **Checkpointing**: Save model weights every N epochs

---

## Part 5: Advanced Training Techniques

### Step 5.1: Learning Rate Scheduling

Learning rate schedules help convergence:

```cpp
class LRScheduler {
protected:
    Optimizer& optimizer_;
    
public:
    LRScheduler(Optimizer& optimizer) : optimizer_(optimizer) {}
    virtual ~LRScheduler() = default;
    
    virtual void step() = 0;
};

class StepLR : public LRScheduler {
private:
    float gamma_;
    int step_size_;
    int current_step_;
    
public:
    StepLR(Optimizer& optimizer, int step_size, float gamma = 0.1f)
        : LRScheduler(optimizer)
        , gamma_(gamma)
        , step_size_(step_size)
        , current_step_(0)
    {}
    
    void step() override {
        current_step_++;
        if (current_step_ % step_size_ == 0) {
            float new_lr = optimizer_.get_learning_rate() * gamma_;
            optimizer_.set_learning_rate(new_lr);
            std::cout << "Learning rate updated to: " << new_lr << "\n";
        }
    }
};
```

**Usage**:

```cpp
nn::Adam optimizer(model.parameters(), 0.1f);
nn::StepLR scheduler(optimizer, step_size=10);

for (int epoch = 0; epoch < 50; ++epoch) {
    train_epoch(...);
    scheduler.step();  // Reduce LR every 10 epochs
}
```

### Step 5.2: Gradient Clipping

Prevents exploding gradients:

```cpp
class Optimizer {
    // ... existing code ...
    
    void clip_grad_norm(float max_norm) {
        // Compute total gradient norm
        float total_norm = 0.0f;
        for (const auto& param : parameters_) {
            if (!param->requires_grad()) continue;
            total_norm += param->grad() * param->grad();
        }
        total_norm = std::sqrt(total_norm);
        
        // Clip if necessary
        if (total_norm > max_norm) {
            float scale = max_norm / total_norm;
            for (auto& param : parameters_) {
                if (!param->requires_grad()) continue;
                param->set_grad(param->grad() * scale);
            }
        }
    }
};
```

**Usage**:

```cpp
optimizer.zero_grad();
loss->backward();
optimizer.clip_grad_norm(1.0f);  // Clip to max norm of 1.0
optimizer.step();
```

### Step 5.3: Regularization (L2 / Weight Decay)

Add L2 penalty to loss:

```cpp
class OptimizerWithWeightDecay : public Optimizer {
protected:
    float weight_decay_;
    
public:
    OptimizerWithWeightDecay(const std::vector<VariablePtr>& params, 
                             float lr, 
                             float weight_decay)
        : Optimizer(params, lr)
        , weight_decay_(weight_decay)
    {}
    
    void step() override {
        for (auto& param : parameters_) {
            if (!param->requires_grad()) continue;
            
            // Add weight decay: grad = grad + λ * weight
            float regularized_grad = param->grad() + weight_decay_ * param->data();
            
            // Update
            float new_data = param->data() - learning_rate_ * regularized_grad;
            param->set_data(new_data);
        }
    }
};
```

**Why weight decay?** It encourages smaller weights, which often helps generalization.

---

## Part 6: Metrics and Evaluation

### Step 6.1: Accuracy for Classification

```cpp
class Metrics {
public:
    static float accuracy(const std::vector<VariablePtr>& predictions,
                         const std::vector<VariablePtr>& targets,
                         float threshold = 0.5f) {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Size mismatch");
        }
        
        size_t correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float pred_class = predictions[i]->data() >= threshold ? 1.0f : 0.0f;
            float true_class = targets[i]->data();
            
            if (std::abs(pred_class - true_class) < 1e-6f) {
                correct++;
            }
        }
        
        return static_cast<float>(correct) / static_cast<float>(predictions.size());
    }
    
    static float mean_absolute_error(const std::vector<VariablePtr>& predictions,
                                     const std::vector<VariablePtr>& targets) {
        float total_error = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            total_error += std::abs(predictions[i]->data() - targets[i]->data());
        }
        return total_error / static_cast<float>(predictions.size());
    }
};
```

### Step 6.2: Confusion Matrix

For binary classification:

```cpp
struct ConfusionMatrix {
    size_t true_positives = 0;
    size_t true_negatives = 0;
    size_t false_positives = 0;
    size_t false_negatives = 0;
    
    void update(float prediction, float target, float threshold = 0.5f) {
        bool pred_positive = prediction >= threshold;
        bool true_positive = target >= threshold;
        
        if (pred_positive && true_positive) true_positives++;
        else if (!pred_positive && !true_positive) true_negatives++;
        else if (pred_positive && !true_positive) false_positives++;
        else false_negatives++;
    }
    
    float precision() const {
        if (true_positives + false_positives == 0) return 0.0f;
        return static_cast<float>(true_positives) / 
               static_cast<float>(true_positives + false_positives);
    }
    
    float recall() const {
        if (true_positives + false_negatives == 0) return 0.0f;
        return static_cast<float>(true_positives) / 
               static_cast<float>(true_positives + false_negatives);
    }
    
    float f1_score() const {
        float p = precision();
        float r = recall();
        if (p + r == 0.0f) return 0.0f;
        return 2.0f * p * r / (p + r);
    }
    
    void print() const {
        std::cout << "Confusion Matrix:\n";
        std::cout << "TP: " << true_positives << ", TN: " << true_negatives << "\n";
        std::cout << "FP: " << false_positives << ", FN: " << false_negatives << "\n";
        std::cout << "Precision: " << precision() << "\n";
        std::cout << "Recall: " << recall() << "\n";
        std::cout << "F1 Score: " << f1_score() << "\n";
    }
};
```

---

## Part 7: Debugging and Visualization

### Step 7.1: Gradient Checking

Verify autodiff is correct using numerical gradients:

```cpp
bool gradient_check(std::function<VariablePtr(VariablePtr)> fn,
                   VariablePtr x,
                   float epsilon = 1e-4f,
                   float tolerance = 1e-3f) {
    // Compute analytical gradient
    x->zero_grad();
    auto y = fn(x);
    y->backward();
    float analytical_grad = x->grad();
    
    // Compute numerical gradient
    float x_val = x->data();
    
    x->set_data(x_val + epsilon);
    float y_plus = fn(x)->data();
    
    x->set_data(x_val - epsilon);
    float y_minus = fn(x)->data();
    
    float numerical_grad = (y_plus - y_minus) / (2.0f * epsilon);
    
    // Restore original value
    x->set_data(x_val);
    
    // Compare
    float diff = std::abs(analytical_grad - numerical_grad);
    bool passed = diff < tolerance;
    
    std::cout << "Gradient Check:\n";
    std::cout << "  Analytical: " << analytical_grad << "\n";
    std::cout << "  Numerical:  " << numerical_grad << "\n";
    std::cout << "  Difference: " << diff << "\n";
    std::cout << "  Status: " << (passed ? "PASS" : "FAIL") << "\n";
    
    return passed;
}
```

**Usage**:

```cpp
auto x = std::make_shared<nn::Variable>(3.0f, true);
auto fn = [](nn::VariablePtr x) { return x * x + x * 2.0f; };

gradient_check(fn, x);
```

### Step 7.2: Training Curves

Log metrics to plot later:

```cpp
class TrainingLogger {
private:
    std::vector<float> train_losses_;
    std::vector<float> val_losses_;
    std::vector<float> learning_rates_;
    
public:
    void log_train_loss(float loss) {
        train_losses_.push_back(loss);
    }
    
    void log_val_loss(float loss) {
        val_losses_.push_back(loss);
    }
    
    void log_learning_rate(float lr) {
        learning_rates_.push_back(lr);
    }
    
    void save_to_csv(const std::string& filename) const {
        std::ofstream file(filename);
        file << "epoch,train_loss,val_loss,learning_rate\n";
        
        for (size_t i = 0; i < train_losses_.size(); ++i) {
            file << i << "," << train_losses_[i] << ",";
            
            if (i < val_losses_.size()) {
                file << val_losses_[i];
            }
            file << ",";
            
            if (i < learning_rates_.size()) {
                file << learning_rates_[i];
            }
            file << "\n";
        }
    }
};
```

---

## Summary and What's Next

### What You've Built

🎉 **Amazing work!** You now have:

- ✅ Multiple loss functions (MSE, BCE, Cross-Entropy)
- ✅ Multiple optimizers (SGD, Momentum, Adam)
- ✅ Data loading infrastructure
- ✅ Complete training framework
- ✅ Metrics and evaluation tools
- ✅ Debugging utilities

### The Complete Machine Learning Pipeline

```
Data → Dataset → DataLoader → Model → Loss → Optimizer → Trained Model
```

You can now train neural networks end-to-end!

### What's Missing

- Batch normalization
- Dropout for regularization
- More sophisticated architectures (CNNs, RNNs)
- Multi-GPU training
- Mixed precision training
- Model serialization (save/load)

---

## 🎯 Final Project: Train a Multi-Layer Network

**Task**: Build and train a 3-layer neural network on a real dataset.

**Requirements**:

1. Use a real dataset (e.g., Iris classification, Boston housing)
2. Implement a 3-layer architecture with ReLU/sigmoid activations
3. Use Adam optimizer with learning rate scheduling
4. Track training and validation metrics
5. Achieve >90% accuracy (for classification) or R² > 0.8 (for regression)
6. Plot training curves

**Bonus challenges**:

- Implement early stopping
- Add L2 regularization
- Try different architectures and compare results
- Implement k-fold cross-validation

---

## Further Reading

1. **Optimization**:
   - "An overview of gradient descent optimization algorithms" (Sebastian Ruder)
   - Adam paper: "Adam: A Method for Stochastic Optimization"

2. **Training Techniques**:
   - "Practical Recommendations for Gradient-Based Training of Deep Architectures"
   - Batch normalization paper

3. **Regularization**:
   - Dropout paper
   - L1/L2 regularization theory

4. **Learning Rate Schedules**:
   - Cyclical learning rates
   - Cosine annealing

---

**Next Tutorial**: Building Convolutional Neural Networks (CNNs)

Let's tackle image recognition! 🖼️
