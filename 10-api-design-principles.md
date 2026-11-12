# API Design Principles

## Overview

This document outlines the design philosophy and API specifications for a C++ neural network library. The goal is to create an intuitive, type-safe, efficient, and modern C++ interface that balances ease of use with performance.

## Design Philosophy

### Core Principles

**1. Type Safety**

- Leverage C++ type system to catch errors at compile time
- Use templates for dimension checking where possible
- Avoid runtime errors that could be compile-time errors

**2. RAII (Resource Acquisition Is Initialization)**

- Automatic resource management
- No manual memory management for users
- Deterministic cleanup

**3. Zero-Cost Abstractions**

- High-level API should compile to efficient code
- No runtime overhead for abstractions
- Inline-able operations

**4. Expressiveness**

- Code should read naturally
- Intuitive method names
- Fluent interfaces where appropriate

**5. Composability**

- Small, focused components
- Easy to combine in custom ways
- Extensible architecture

### Guiding Questions

For each API decision, ask:

- Is this safe by default?
- Can errors be caught at compile time?
- Is the intent clear from reading the code?
- Does this impose unnecessary runtime cost?
- Can users extend this easily?

## Type System Design

### Tensor Type

**Core Type**:

```
Tensor<T, DimensionPolicy>

Where:
- T: Element type (float, double, int, etc.)
- DimensionPolicy: Compile-time or runtime dimension checking
```

**Example Usage**:

```
// Runtime dimensions (flexible)
Tensor<float> dynamicTensor({224, 224, 3});

// Compile-time dimensions (maximum safety)
Tensor<float, Shape<224, 224, 3>> staticTensor;
```

**Benefits**:

- Type safety
- Clear element type
- Optimization opportunities (static dimensions)

### Shape Representation

**Runtime Shape**:

```
Shape {
    dimensions: vector<size_t>
    
    size_t rank() const
    size_t dim(size_t axis) const
    size_t total_size() const
}
```

**Compile-Time Shape** (template):

```
template<size_t... Dims>
struct StaticShape {
    static constexpr size_t rank = sizeof...(Dims)
    static constexpr array<size_t, rank> dimensions = {Dims...}
}
```

**Shape Operations**:

```
// Shape checking
bool compatible(Shape a, Shape b)
Shape broadcast(Shape a, Shape b)

// Shape manipulation
Shape reshape(Shape original, Shape target)
Shape transpose(Shape original, vector<size_t> axes)
```

## Layer API Design

### Layer Interface

**Concept**: Layers are composable units

**Core Interface**:

```
Layer {
    // Core operations
    virtual Tensor forward(const Tensor& input) = 0
    virtual Tensor backward(const Tensor& gradient) = 0
    
    // Parameter management
    virtual vector<Parameter*> parameters()
    virtual void zero_gradients()
    
    // Persistence
    virtual void save(OutputStream& stream)
    virtual void load(InputStream& stream)
    
    // Metadata
    virtual Shape output_shape(Shape input_shape)
    virtual string name()
}
```

**Parameter Type**:

```
Parameter {
    Tensor value
    Tensor gradient
    
    bool requires_grad = true
    string name
    
    void zero_grad()
}
```

### Layer Types

**Dense (Fully Connected)**:

```
DenseLayer {
    DenseLayer(size_t input_dim, size_t output_dim)
    
    Properties:
    - Parameter weights: (output_dim, input_dim)
    - Parameter bias: (output_dim)
    - Activation: optional
}

Example:
auto dense = DenseLayer(512, 256)
              .activation(ActivationType::ReLU)
              .use_bias(true)
              .initializer(Initializer::He)
```

**Convolutional**:

```
Conv2D {
    Conv2D(
        size_t in_channels,
        size_t out_channels,
        size_t kernel_size,
        size_t stride = 1,
        size_t padding = 0
    )
    
    Properties:
    - Parameter weights: (out_channels, in_channels, kernel_h, kernel_w)
    - Parameter bias: (out_channels)
}

Example:
auto conv = Conv2D(3, 64, 3)  // 3→64 channels, 3x3 kernel
             .stride(1)
             .padding(1)
             .initializer(Initializer::He)
```

**Activation**:

```
ActivationLayer {
    ActivationLayer(ActivationType type)
}

Types:
- ReLU
- Sigmoid
- Tanh
- LeakyReLU
- ELU
- GELU
- Softmax

Example:
auto activation = ActivationLayer(ActivationType::ReLU)
```

**Dropout**:

```
Dropout {
    Dropout(float rate)
    
    void set_training(bool training)  // Mode control
}

Example:
auto dropout = Dropout(0.5)
```

**BatchNormalization**:

```
BatchNorm {
    BatchNorm(size_t num_features)
    
    Properties:
    - Parameter gamma: (num_features) [learnable]
    - Parameter beta: (num_features) [learnable]
    - Buffer running_mean: (num_features)
    - Buffer running_var: (num_features)
    
    void set_training(bool training)
}
```

### Layer Builder Pattern

**Fluent Interface**:

```
auto layer = DenseLayer(512, 256)
             .activation(ActivationType::ReLU)
             .use_bias(true)
             .initializer(Initializer::He)
             .regularizer(L2Regularizer(0.001))
             .name("hidden_layer_1")
```

**Benefits**:

- Readable configuration
- Optional parameters clear
- Type-safe
- Extensible

## Model API Design

### Sequential Model

**Concept**: Linear stack of layers

**API**:

```
Sequential {
    Sequential& add(unique_ptr<Layer> layer)
    
    template<typename LayerType, typename... Args>
    Sequential& add(Args&&... args)  // Construct in-place
    
    Tensor forward(const Tensor& input)
    Tensor backward(const Tensor& gradient)
    
    vector<Parameter*> parameters()
}

Example:
Sequential model;
model.add<DenseLayer>(784, 256)
     .add<ActivationLayer>(ActivationType::ReLU)
     .add<Dropout>(0.5)
     .add<DenseLayer>(256, 128)
     .add<ActivationLayer>(ActivationType::ReLU)
     .add<DenseLayer>(128, 10)
     .add<ActivationLayer>(ActivationType::Softmax)
```

### Functional Model

**Concept**: Arbitrary computation graphs

**API**:

```
// Define layers
auto input = Input({224, 224, 3})
auto conv1 = Conv2D(3, 64, 3)(input)
auto relu1 = ReLU()(conv1)
auto pool1 = MaxPool2D(2)(relu1)
auto conv2 = Conv2D(64, 128, 3)(pool1)
auto relu2 = ReLU()(conv2)
auto flatten = Flatten()(relu2)
auto dense = DenseLayer(flatten_size, 10)(flatten)
auto output = Softmax()(dense)

// Create model
Model model(input, output)
```

**Benefits**:

- Supports skip connections
- Multiple inputs/outputs
- Arbitrary graphs
- Clear data flow

## Loss Function API

### Loss Interface

**Core Interface**:

```
Loss {
    virtual float compute(const Tensor& predictions, const Tensor& targets) = 0
    virtual Tensor gradient(const Tensor& predictions, const Tensor& targets) = 0
    
    virtual string name() = 0
}
```

### Loss Types

**Classification Losses**:

```
CrossEntropyLoss {
    CrossEntropyLoss()
    
    // Numerically stable: operates on logits
    float compute(const Tensor& logits, const Tensor& targets)
}

BinaryCrossEntropyLoss {
    BinaryCrossEntropyLoss()
}
```

**Regression Losses**:

```
MSELoss {}
MAELoss {}
HuberLoss {
    HuberLoss(float delta = 1.0)
}
```

**Example Usage**:

```
auto loss_fn = CrossEntropyLoss()
float loss = loss_fn.compute(predictions, targets)
Tensor grad = loss_fn.gradient(predictions, targets)
```

## Optimizer API

### Optimizer Interface

**Core Interface**:

```
Optimizer {
    Optimizer(vector<Parameter*> parameters, float learning_rate)
    
    virtual void step() = 0
    virtual void zero_grad()
    
    void set_learning_rate(float lr)
    float get_learning_rate() const
}
```

### Optimizer Types

**SGD**:

```
SGD {
    SGD(
        vector<Parameter*> params,
        float lr,
        float momentum = 0.0,
        float weight_decay = 0.0
    )
}

Example:
auto optimizer = SGD(model.parameters(), 0.01)
                  .momentum(0.9)
                  .weight_decay(0.0001)
```

**Adam**:

```
Adam {
    Adam(
        vector<Parameter*> params,
        float lr = 0.001,
        float beta1 = 0.9,
        float beta2 = 0.999,
        float epsilon = 1e-8
    )
}

Example:
auto optimizer = Adam(model.parameters())
```

**AdamW**:

```
AdamW {
    // Adam with decoupled weight decay
    AdamW(
        vector<Parameter*> params,
        float lr = 0.001,
        float weight_decay = 0.01
    )
}
```

### Learning Rate Schedulers

**Scheduler Interface**:

```
LRScheduler {
    LRScheduler(Optimizer& optimizer)
    
    virtual void step(size_t epoch) = 0
    virtual float get_lr() = 0
}
```

**Types**:

```
StepLR {
    StepLR(Optimizer& opt, size_t step_size, float gamma = 0.1)
}

ExponentialLR {
    ExponentialLR(Optimizer& opt, float gamma)
}

CosineAnnealingLR {
    CosineAnnealingLR(Optimizer& opt, size_t T_max, float eta_min = 0)
}

Example:
auto scheduler = CosineAnnealingLR(optimizer, 100)
for (size_t epoch = 0; epoch < 100; ++epoch) {
    // Training...
    scheduler.step(epoch)
}
```

## Training Loop API

### High-Level Training API

**Simple Interface**:

```
Trainer {
    Trainer(Model& model, Loss& loss, Optimizer& optimizer)
    
    TrainingResult fit(
        DataLoader& train_data,
        DataLoader& val_data,
        size_t epochs,
        TrainingConfig config = {}
    )
    
    EvaluationResult evaluate(DataLoader& data)
}

TrainingConfig {
    bool early_stopping = false
    size_t patience = 10
    bool verbose = true
    size_t checkpoint_frequency = 0
    string checkpoint_path
}

Example:
Trainer trainer(model, CrossEntropyLoss(), Adam(model.parameters()))
auto results = trainer.fit(train_loader, val_loader, 100)
```

### Manual Training Loop

**For Custom Logic**:

```
for (size_t epoch = 0; epoch < epochs; ++epoch) {
    model.train()  // Set training mode
    
    for (auto& batch : train_loader) {
        auto [inputs, targets] = batch
        
        // Forward pass
        auto outputs = model.forward(inputs)
        float loss = loss_fn.compute(outputs, targets)
        
        // Backward pass
        optimizer.zero_grad()
        auto grad = loss_fn.gradient(outputs, targets)
        model.backward(grad)
        
        // Update
        optimizer.step()
    }
    
    // Validation
    model.eval()  // Set evaluation mode
    float val_loss = evaluate(model, val_loader, loss_fn)
    
    // Learning rate schedule
    scheduler.step(epoch)
}
```

## Data Loading API

### DataLoader

**Concept**: Efficient data batching and iteration

**Interface**:

```
DataLoader {
    DataLoader(
        Dataset& dataset,
        size_t batch_size,
        bool shuffle = true,
        size_t num_workers = 1  // Parallel loading
    )
    
    Iterator begin()
    Iterator end()
    
    size_t size() const  // Number of batches
}

Dataset {
    virtual size_t size() const = 0
    virtual Tensor operator[](size_t index) const = 0
    virtual pair<Tensor, Tensor> get(size_t index) const = 0
}

Example:
auto loader = DataLoader(dataset, 32).shuffle(true)

for (auto& [inputs, targets] : loader) {
    // Training batch
}
```

### Dataset Types

**TensorDataset**:

```
TensorDataset {
    TensorDataset(Tensor inputs, Tensor targets)
}
```

**ImageDataset**:

```
ImageDataset {
    ImageDataset(
        vector<string> image_paths,
        vector<int> labels,
        Transform transform = nullptr
    )
}
```

## Serialization API

### Model Save/Load

**Simple API**:

```
// Save
model.save("model.bin")
model.save_state_dict("weights.bin")  // Weights only

// Load
Model model = Model::load("model.bin")
model.load_state_dict("weights.bin")  // Load weights only
```

**State Dictionary**:

```
StateDict {
    map<string, Tensor> parameters
    map<string, Tensor> buffers
    
    void save(const string& path)
    static StateDict load(const string& path)
}

auto state = model.state_dict()
state.save("checkpoint.bin")

// Later
model.load_state_dict(StateDict::load("checkpoint.bin"))
```

## Error Handling

### Exception Hierarchy

```
NeuralNetException  // Base exception
├── ShapeError  // Shape mismatch errors
├── ValueError  // Invalid values
├── TypeError  // Type errors
├── ConfigurationError  // Invalid configuration
└── ComputationError  // Runtime computation errors
```

### Usage

**Dimension Mismatch**:

```
Shape expected = {10, 256}
Shape actual = {10, 128}

if (expected != actual) {
    throw ShapeError(
        "Expected shape " + to_string(expected) +
        " but got " + to_string(actual)
    )
}
```

**Validation**:

```
if (learning_rate <= 0) {
    throw ValueError("Learning rate must be positive")
}
```

## Thread Safety

### Guarantees

**Thread-Safe Operations**:

- Forward pass (read-only)
- Multiple concurrent inferences
- DataLoader with num_workers > 1

**Not Thread-Safe**:

- Backward pass
- Parameter updates
- Training

**Recommendation**: Use external synchronization for training

## Memory Management

### Smart Pointers

**Ownership**:

```
// Library owns layers
unique_ptr<Layer> layer = make_unique<DenseLayer>(512, 256)
model.add(std::move(layer))

// Shared ownership when needed
shared_ptr<Tensor> shared_tensor = ...
```

**Parameters**:

```
// Library manages parameter memory
// Users work with references/pointers
vector<Parameter*> params = model.parameters()  // Non-owning
```

### Tensor Memory

**Reference Semantics**:

```
Tensor a = {1, 2, 3, 4}
Tensor b = a  // Shallow copy (shared data)
Tensor c = a.clone()  // Deep copy
```

**In-Place Operations**:

```
tensor.relu_()  // In-place (trailing underscore convention)
tensor.add_(other)
```

## C++ Best Practices

### Modern C++ Features

**Use**:

- `auto` for type deduction
- Range-based for loops
- Smart pointers
- Move semantics
- Template metaprogramming (where beneficial)
- `constexpr` for compile-time computation
- Structured bindings

**Example**:

```
// Good
for (const auto& batch : data_loader) {
    auto [inputs, targets] = batch
    auto output = model.forward(inputs)
}

// Avoid
for (size_t i = 0; i < data_loader.size(); ++i) {
    Batch batch = data_loader[i]
    Tensor inputs = batch.first
    Tensor targets = batch.second
}
```

### Naming Conventions

**Classes**: PascalCase

```
class DenseLayer
class Adam
class Sequential
```

**Methods/Functions**: snake_case

```
forward()
zero_grad()
get_learning_rate()
```

**Constants**: UPPER_SNAKE_CASE

```
const float DEFAULT_LEARNING_RATE = 0.001
```

**Private Members**: trailing_underscore_

```
private:
    Tensor weights_
    size_t input_dim_
```

## API Examples (Specification)

### Example 1: Image Classification

```
// Define model
Sequential model
model.add<Conv2D>(3, 32, 3).padding(1)
     .add<ReLU>()
     .add<MaxPool2D>(2)
     .add<Conv2D>(32, 64, 3).padding(1)
     .add<ReLU>()
     .add<MaxPool2D>(2)
     .add<Flatten>()
     .add<DenseLayer>(64 * 56 * 56, 128)
     .add<ReLU>()
     .add<Dropout>(0.5)
     .add<DenseLayer>(128, 10)

// Prepare training
auto loss = CrossEntropyLoss()
auto optimizer = Adam(model.parameters(), 0.001)

// Train
Trainer trainer(model, loss, optimizer)
trainer.fit(train_loader, val_loader, 50)

// Evaluate
auto results = trainer.evaluate(test_loader)
```

### Example 2: Custom Training Loop

```
model.train()

for (size_t epoch = 0; epoch < 100; ++epoch) {
    float epoch_loss = 0.0
    
    for (const auto& [inputs, targets] : train_loader) {
        // Forward
        auto predictions = model.forward(inputs)
        float loss = loss_fn.compute(predictions, targets)
        
        // Backward
        optimizer.zero_grad()
        auto grad_output = loss_fn.gradient(predictions, targets)
        model.backward(grad_output)
        
        // Clip gradients
        clip_grad_norm(model.parameters(), 5.0)
        
        // Update
        optimizer.step()
        
        epoch_loss += loss
    }
    
    cout << "Epoch " << epoch << " Loss: " << epoch_loss << endl
}
```

## Summary

### API Design Priorities

1. **Safety**: Catch errors early (compile-time > runtime)
2. **Clarity**: Code reads like the intent
3. **Performance**: Zero-cost abstractions
4. **Flexibility**: Support both simple and advanced use cases
5. **Modern**: Use C++17/20 features appropriately

### Key Decisions

- RAII for all resources
- Template-based type safety where beneficial
- Fluent interfaces for configuration
- Clear ownership semantics
- Exceptions for error handling
- Smart pointers for memory management

---

**Previous**: [Training Strategies](./09-training-strategies.md)  
**Next**: [Library Architecture](./11-library-architecture.md)
