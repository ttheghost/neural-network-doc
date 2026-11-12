# Machine Learning in C: Regression and Data Preparation

## Overview

When building machine learning systems like regression models from scratch in C, the complexity often doesn't lie in the core algorithms (like gradient descent), but rather in **data manipulation, memory management, and data preparation**.

For tasks strictly dealing with regression and tabular data, we don't need highly complex n-dimensional tensor libraries. A robust 2D `Table` (or `Matrix`) structure is sufficient to represent our features ($X$), targets ($y$), weights ($W$), and intermediate statistics (means, standard deviations).

This document covers:
1. **Design Decisions & QoL**: Designing a robust `Table` struct tailored for ML data.
2. **Table Utilities**: Slicing, copying, filling, row shuffling, train/test splitting.
3. **Data Preparation**: Mean, standard deviation, min/max, Z-score, min-max normalization.
4. **Linear Regression Theory**: The hypothesis, MSE cost function, and gradient derivation.
5. **Optimization**: Batch GD, Stochastic GD, and Mini-Batch GD.
6. **Evaluation Metrics**: MAE, RMSE, R² score.
7. **Polynomial Regression**: Extending linear regression with feature engineering.
8. **CSV Loading**: Parsing real data files.
9. **Complete Worked Example**: End-to-end training on synthetic data.

---

## 1. Design Decisions and the `Table` Data Structure

### 1.1 Why a Single Struct?

In Python you have NumPy arrays, Pandas DataFrames, and scalars — all interoperable. In C, every type boundary is a function-signature headache. Our key design decision: **one struct for everything**.

| What               | Representation          | Example Shape |
|---------------------|-------------------------|---------------|
| Full dataset        | `Table(m, n)`           | `(1000, 5)`  |
| Feature matrix X    | `Table(m, n_features)`  | `(1000, 4)`  |
| Target vector y     | `Table(m, 1)`           | `(1000, 1)`  |
| Weight vector W     | `Table(n_features, 1)`  | `(4, 1)`     |
| Column means        | `Table(1, n_features)`  | `(1, 4)`     |
| A single scalar     | `Table(1, 1)`           | `(1, 1)`     |

This means `table_mean_axis0` returns a `Table`, `table_extract_column` returns a `Table`, and the gradient function takes/returns `Table*`. No mental context switching.

### 1.2 Memory Layout

We use **row-major order**: element `[i][j]` is stored at flat index `i * cols + j`. This means iterating across columns within a row is cache-friendly (contiguous memory access).

```
Conceptual Table (3 rows, 4 cols):
  row 0: [  1.0   2.0   3.0   4.0 ]
  row 1: [  5.0   6.0   7.0   8.0 ]
  row 2: [  9.0  10.0  11.0  12.0 ]

Flat memory (row-major):
  [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 ]
    ^--- row 0 ---^  ^--- row 1 ---^  ^--- row 2 ----------^
```

### 1.3 Core Struct and Lifecycle

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ================================================================
 * THE TABLE: Our universal data structure for ML regression
 * ================================================================ */
typedef struct {
    float *data;   /* Heap-allocated, row-major contiguous storage */
    int    rows;   /* Number of rows    (m = samples in a dataset) */
    int    cols;   /* Number of columns (n = features)             */
} Table;

/* ---------- Allocation ---------- */

/*
 * Allocates a Table of the given shape, zero-initialized.
 * calloc guarantees all bytes are 0, so all floats are 0.0f.
 */
Table table_alloc(int rows, int cols) {
    Table t;
    t.rows = rows;
    t.cols = cols;
    t.data = (float *)calloc((size_t)rows * cols, sizeof(float));
    if (!t.data) {
        fprintf(stderr, "FATAL: table_alloc(%d, %d) — out of memory\n", rows, cols);
        exit(EXIT_FAILURE);
    }
    return t;
}

/*
 * Allocates AND fills every element with a constant value.
 * Useful for initializing biases to 0.0 or weights to a small constant.
 */
Table table_alloc_fill(int rows, int cols, float fill_value) {
    Table t = table_alloc(rows, cols);
    int total = rows * cols;
    for (int i = 0; i < total; i++) {
        t.data[i] = fill_value;
    }
    return t;
}

/* ---------- Deallocation ---------- */

/*
 * Frees the table's data and zeroes out the metadata.
 * Safe to call multiple times (checks for NULL).
 * After this call the Table is in a "dead" state and must not be used
 * unless re-allocated.
 */
void table_free(Table *t) {
    if (t && t->data) {
        free(t->data);
        t->data = NULL;
    }
    if (t) {
        t->rows = 0;
        t->cols = 0;
    }
}

/* ---------- Element Access ---------- */

/*
 * These are static inline so the compiler inlines them — no function-call
 * overhead in tight loops (critical for gradient computation).
 */
static inline float table_get(const Table *t, int row, int col) {
    return t->data[row * t->cols + col];
}

static inline void table_set(Table *t, int row, int col, float val) {
    t->data[row * t->cols + col] = val;
}

/* Flat index access (when treating the table as a 1D buffer) */
static inline float table_flat_get(const Table *t, int idx) {
    return t->data[idx];
}

static inline void table_flat_set(Table *t, int idx, float val) {
    t->data[idx] = val;
}
```

---

## 2. Table Utilities (QoL)

### 2.1 Deep Copy

```c
/*
 * Creates an independent deep copy. Useful when you need to normalize
 * a copy of X while keeping the original intact for later inspection.
 */
Table table_copy(const Table *src) {
    Table dst = table_alloc(src->rows, src->cols);
    memcpy(dst.data, src->data, (size_t)src->rows * src->cols * sizeof(float));
    return dst;
}
```

### 2.2 Printing / Inspection

```c
/*
 * Prints the first `n` rows (or all if fewer exist).
 * Mimics pandas DataFrame.head().
 */
void table_print_head(const Table *t, int n, const char *name) {
    int show = (n < t->rows) ? n : t->rows;
    printf("--- %s (%d rows × %d cols, showing %d) ---\n",
           name, t->rows, t->cols, show);
    for (int i = 0; i < show; i++) {
        printf("  [%4d] ", i);
        for (int j = 0; j < t->cols; j++) {
            printf("%10.4f ", table_get(t, i, j));
        }
        printf("\n");
    }
    if (t->rows > show) printf("  ... (%d more rows)\n", t->rows - show);
    printf("\n");
}

/*
 * Prints shape info only — useful in debug logging.
 */
void table_print_shape(const Table *t, const char *name) {
    printf("%s: shape=(%d, %d), total_elements=%d\n",
           name, t->rows, t->cols, t->rows * t->cols);
}
```

### 2.3 Column and Row Extraction

```c
/*
 * Extracts column `col_idx` into a new (m × 1) Table.
 * Use case: splitting target column from a combined dataset.
 *
 *   Table y = table_extract_column(&dataset, 4);  // target is col 4
 */
Table table_extract_column(const Table *t, int col_idx) {
    Table col = table_alloc(t->rows, 1);
    for (int i = 0; i < t->rows; i++) {
        col.data[i] = table_get(t, i, col_idx);
    }
    return col;
}

/*
 * Extracts columns [col_start, col_end) into a new Table.
 * Use case: extracting features from a dataset where the last col is the target.
 *
 *   Table X = table_extract_columns(&dataset, 0, 4);  // cols 0,1,2,3
 */
Table table_extract_columns(const Table *t, int col_start, int col_end) {
    int new_cols = col_end - col_start;
    Table sub = table_alloc(t->rows, new_cols);
    for (int i = 0; i < t->rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            table_set(&sub, i, j, table_get(t, i, col_start + j));
        }
    }
    return sub;
}

/*
 * Extracts rows [row_start, row_end) into a new Table.
 * Use case: manual train/test split.
 */
Table table_extract_rows(const Table *t, int row_start, int row_end) {
    int new_rows = row_end - row_start;
    Table sub = table_alloc(new_rows, t->cols);
    memcpy(sub.data,
           t->data + row_start * t->cols,
           (size_t)new_rows * t->cols * sizeof(float));
    return sub;
}
```

### 2.4 Row Shuffling (Fisher-Yates)

**Why shuffle?** SGD assumes samples are drawn i.i.d. (independently and identically distributed). If your CSV is sorted by target value (e.g., all cheap houses first, expensive houses last), the gradient updates in one epoch will be systematically biased. Shuffling before each epoch fixes this.

**Algorithm**: Fisher-Yates shuffle — iterate from the last row to the first, swapping each row with a randomly chosen row at or before it. This produces a uniformly random permutation in O(m) time.

```c
/*
 * Shuffles rows of X and y in unison (same permutation applied to both).
 * This is critical: if we shuffle X without applying the same shuffle to y,
 * we break the correspondence between features and targets.
 *
 * Must call srand(time(NULL)) once before using this.
 */
void table_shuffle_rows_paired(Table *X, Table *y) {
    int m = X->rows;
    int x_cols = X->cols;
    int y_cols = y->cols;

    /* Temporary buffers for one row swap */
    float *tmp_x = (float *)malloc(x_cols * sizeof(float));
    float *tmp_y = (float *)malloc(y_cols * sizeof(float));

    for (int i = m - 1; i > 0; i--) {
        int j = rand() % (i + 1);  /* random index in [0, i] */

        /* Swap row i and row j in X */
        memcpy(tmp_x, X->data + i * x_cols, x_cols * sizeof(float));
        memcpy(X->data + i * x_cols, X->data + j * x_cols, x_cols * sizeof(float));
        memcpy(X->data + j * x_cols, tmp_x, x_cols * sizeof(float));

        /* Swap row i and row j in y */
        memcpy(tmp_y, y->data + i * y_cols, y_cols * sizeof(float));
        memcpy(y->data + i * y_cols, y->data + j * y_cols, y_cols * sizeof(float));
        memcpy(y->data + j * y_cols, tmp_y, y_cols * sizeof(float));
    }

    free(tmp_x);
    free(tmp_y);
}
```

### 2.5 Train / Test Split

**Why split?** We evaluate on unseen data to check if the model generalizes or just memorizes the training set (overfitting). A typical split is 80% train / 20% test.

**Important**: Shuffle before splitting. Otherwise, if the data is ordered, the test set will be an unrepresentative slice.

```c
/*
 * Splits a dataset into train and test portions.
 * train_ratio is typically 0.8 (80% train).
 *
 * Usage:
 *   Table X_train, X_test, y_train, y_test;
 *   table_train_test_split(&X, &y, 0.8f, &X_train, &X_test, &y_train, &y_test);
 */
void table_train_test_split(
    const Table *X, const Table *y, float train_ratio,
    Table *X_train, Table *X_test,
    Table *y_train, Table *y_test
) {
    int m = X->rows;
    int split = (int)(m * train_ratio);

    *X_train = table_extract_rows(X, 0, split);
    *X_test  = table_extract_rows(X, split, m);
    *y_train = table_extract_rows(y, 0, split);
    *y_test  = table_extract_rows(y, split, m);
}
```

---

## 3. Data Preparation: Math & Implementation

### 3.1 Why Normalize?

Consider two features:
- $x_1$: square footage, range [500, 5000]
- $x_2$: number of bedrooms, range [1, 6]

The gradient for $w_1$ (associated with $x_1$) will be **~1000× larger** than the gradient for $w_2$ because the gradient formula multiplies the error by the feature value:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_i (\hat{y}_i - y_i) \cdot x_{i,j}$$

This creates an elongated loss landscape where gradient descent zigzags instead of heading straight toward the minimum. Normalization makes the landscape roughly circular, enabling faster and more stable convergence.

### 3.2 Column-Wise Mean

**Math**: The arithmetic mean of column $j$ across all $m$ rows:

$$\mu_j = \frac{1}{m} \sum_{i=0}^{m-1} x_{i,j}$$

**Worked Example**:
```
X (4 rows × 3 cols):
  [  2.0   100.0   1.0 ]
  [  4.0   200.0   3.0 ]
  [  6.0   300.0   5.0 ]
  [  8.0   400.0   7.0 ]

mean_col0 = (2+4+6+8)/4 = 5.0
mean_col1 = (100+200+300+400)/4 = 250.0
mean_col2 = (1+3+5+7)/4 = 4.0

Result: Table(1,3) = [ 5.0, 250.0, 4.0 ]
```

```c
/*
 * Returns a (1 × n) Table where each element is the mean of that column.
 * Accumulates in double to avoid precision loss for large datasets.
 */
Table table_mean_axis0(const Table *X) {
    Table mean = table_alloc(1, X->cols);
    for (int j = 0; j < X->cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < X->rows; i++) {
            sum += (double)table_get(X, i, j);
        }
        mean.data[j] = (float)(sum / X->rows);
    }
    return mean;
}
```

### 3.3 Column-Wise Row Mean (axis=1)

For completeness — returns an $(m \times 1)$ Table where each element is the mean across that row's columns.

```c
Table table_mean_axis1(const Table *X) {
    Table mean = table_alloc(X->rows, 1);
    for (int i = 0; i < X->rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < X->cols; j++) {
            sum += (double)table_get(X, i, j);
        }
        mean.data[i] = (float)(sum / X->cols);
    }
    return mean;
}
```

### 3.4 Column-Wise Standard Deviation

**Math**: Population standard deviation of column $j$:

$$\sigma_j = \sqrt{\frac{1}{m} \sum_{i=0}^{m-1} (x_{i,j} - \mu_j)^2}$$

**Worked Example** (continuing from above):
```
col 0: values = [2, 4, 6, 8], mean = 5.0
  deviations² = (2-5)²+(4-5)²+(6-5)²+(8-5)² = 9+1+1+9 = 20
  σ = √(20/4) = √5 ≈ 2.236

col 1: values = [100, 200, 300, 400], mean = 250.0
  deviations² = 150²+50²+50²+150² = 22500+2500+2500+22500 = 50000
  σ = √(50000/4) = √12500 ≈ 111.803

col 2: values = [1, 3, 5, 7], mean = 4.0
  deviations² = 9+1+1+9 = 20
  σ = √(20/4) = √5 ≈ 2.236

Result: Table(1,3) = [ 2.236, 111.803, 2.236 ]
```

```c
/*
 * Returns (1 × n) Table of per-column population standard deviations.
 * Requires pre-computed means (to avoid recomputing).
 */
Table table_stddev_axis0(const Table *X, const Table *mean) {
    Table sd = table_alloc(1, X->cols);
    for (int j = 0; j < X->cols; j++) {
        double sum_sq = 0.0;
        float mu = mean->data[j];
        for (int i = 0; i < X->rows; i++) {
            double diff = (double)table_get(X, i, j) - (double)mu;
            sum_sq += diff * diff;
        }
        sd.data[j] = (float)sqrt(sum_sq / X->rows);
    }
    return sd;
}
```

### 3.5 Column-Wise Min and Max

**Math**:
$$\text{min}_j = \min_i(x_{i,j}), \quad \text{max}_j = \max_i(x_{i,j})$$

```c
Table table_min_axis0(const Table *X) {
    Table mn = table_alloc(1, X->cols);
    for (int j = 0; j < X->cols; j++) {
        float m = table_get(X, 0, j);
        for (int i = 1; i < X->rows; i++) {
            float v = table_get(X, i, j);
            if (v < m) m = v;
        }
        mn.data[j] = m;
    }
    return mn;
}

Table table_max_axis0(const Table *X) {
    Table mx = table_alloc(1, X->cols);
    for (int j = 0; j < X->cols; j++) {
        float m = table_get(X, 0, j);
        for (int i = 1; i < X->rows; i++) {
            float v = table_get(X, i, j);
            if (v > m) m = v;
        }
        mx.data[j] = m;
    }
    return mx;
}
```

### 3.6 Z-Score Normalization (Standardization)

**Math**: Transforms each feature to have mean=0 and std=1:

$$x'_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j + \epsilon}$$

**Worked Example** (col 0 from above, μ=5.0, σ≈2.236):
```
x'[0] = (2.0-5.0)/2.236 = -1.342
x'[1] = (4.0-5.0)/2.236 = -0.447
x'[2] = (6.0-5.0)/2.236 =  0.447
x'[3] = (8.0-5.0)/2.236 =  1.342
```

After normalization: mean ≈ 0.0, std ≈ 1.0 ✓

```c
/*
 * Normalizes X in-place using pre-computed mean and stddev.
 * epsilon prevents division by zero for constant columns.
 *
 * CRITICAL: Save mean and stddev after fitting on training data.
 * When predicting on new data, apply the SAME mean/stddev —
 * never recompute from the test set (that causes data leakage).
 */
void table_normalize_zscore(Table *X, const Table *mean, const Table *stddev) {
    const float eps = 1e-8f;
    for (int j = 0; j < X->cols; j++) {
        float mu = mean->data[j];
        float sigma = stddev->data[j] + eps;
        for (int i = 0; i < X->rows; i++) {
            float val = table_get(X, i, j);
            table_set(X, i, j, (val - mu) / sigma);
        }
    }
}
```

### 3.7 Min-Max Normalization

**Math**: Scales each feature to the range [0, 1]:

$$x'_{i,j} = \frac{x_{i,j} - \text{min}_j}{\text{max}_j - \text{min}_j + \epsilon}$$

**When to use Z-score vs Min-Max**:
- **Z-score**: When features are roughly normally distributed, or when outliers should retain their relative distance. Preferred for regression with gradient descent.
- **Min-Max**: When you need bounded outputs (e.g., pixel values in [0,1]), or when the distribution is uniform-ish.

```c
void table_normalize_minmax(Table *X, const Table *min_vals, const Table *max_vals) {
    const float eps = 1e-8f;
    for (int j = 0; j < X->cols; j++) {
        float lo = min_vals->data[j];
        float range = max_vals->data[j] - lo + eps;
        for (int i = 0; i < X->rows; i++) {
            float val = table_get(X, i, j);
            table_set(X, i, j, (val - lo) / range);
        }
    }
}
```

### 3.8 Denormalization (Inverse Transform)

After training, the model's predictions are in normalized space. To get real-world values back, we invert the transform.

```c
/* Inverse of Z-score: x = x'*σ + μ */
void table_denormalize_zscore(Table *X, const Table *mean, const Table *stddev) {
    for (int j = 0; j < X->cols; j++) {
        float mu = mean->data[j];
        float sigma = stddev->data[j];
        for (int i = 0; i < X->rows; i++) {
            float val = table_get(X, i, j);
            table_set(X, i, j, val * sigma + mu);
        }
    }
}

/* Inverse of Min-Max: x = x'*(max-min) + min */
void table_denormalize_minmax(Table *X, const Table *min_vals, const Table *max_vals) {
    for (int j = 0; j < X->cols; j++) {
        float lo = min_vals->data[j];
        float range = max_vals->data[j] - lo;
        for (int i = 0; i < X->rows; i++) {
            float val = table_get(X, i, j);
            table_set(X, i, j, val * range + lo);
        }
    }
}
```

---

## 4. Linear Regression: Theory

### 4.1 The Hypothesis

A linear regression model predicts a continuous target value $\hat{y}$ as a weighted sum of input features plus a bias:

$$\hat{y}_i = w_0 x_{i,0} + w_1 x_{i,1} + \cdots + w_{n-1} x_{i,n-1} + b$$

In matrix form for all $m$ samples at once:

$$\hat{\mathbf{y}} = X \mathbf{w} + b$$

Where:
- $X$ is $(m \times n)$ — the feature matrix
- $\mathbf{w}$ is $(n \times 1)$ — the weight vector
- $b$ is a scalar — the bias (also called the intercept)
- $\hat{\mathbf{y}}$ is $(m \times 1)$ — the predictions

**Geometric Interpretation**: We're fitting a hyperplane in $n$-dimensional space that best approximates the target values. For 1 feature, it's a line ($y = wx + b$). For 2 features, it's a plane. For $n$ features, it's an $n$-dimensional hyperplane.

```c
/*
 * predict_linear: ŷ = Xw + b
 *
 * X is (m × n), W is (n × 1), bias is scalar.
 * Returns a new (m × 1) Table of predictions.
 *
 * NOTE: We accumulate the dot product in double precision.
 * With 1000 features and float32 inputs, single-precision
 * accumulation can lose 2-3 digits of accuracy.
 */
Table predict_linear(const Table *X, const Table *W, float bias) {
    Table y_pred = table_alloc(X->rows, 1);

    for (int i = 0; i < X->rows; i++) {
        double sum = (double)bias;
        for (int j = 0; j < X->cols; j++) {
            sum += (double)table_get(X, i, j) * (double)W->data[j];
        }
        y_pred.data[i] = (float)sum;
    }
    return y_pred;
}
```

### 4.2 Cost Function: Mean Squared Error (MSE)

**Math**: MSE measures the average squared difference between predictions and targets:

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=0}^{m-1} (\hat{y}_i - y_i)^2$$

**Why the ½?** When we take the derivative, the 2 from the exponent cancels the ½, giving a cleaner gradient formula. This is purely a convenience — it doesn't change where the minimum is.

**Why squared (not absolute)?** Three reasons:
1. **Differentiable everywhere** — absolute value has a kink at 0
2. **Penalizes large errors more** — a prediction off by 10 is penalized 100×, not 10×
3. **Has a unique global minimum** — the loss surface is convex for linear regression

**Worked Example**:
```
y_pred = [2.5, 0.0, 2.1]
y_true = [3.0, -0.5, 2.0]

errors     = [2.5-3.0, 0.0-(-0.5), 2.1-2.0] = [-0.5, 0.5, 0.1]
errors²    = [0.25, 0.25, 0.01]
sum        = 0.51
MSE (1/2m) = 0.51 / (2*3) = 0.085
```

```c
float calculate_mse(const Table *y_pred, const Table *y) {
    double sum_sq = 0.0;
    for (int i = 0; i < y->rows; i++) {
        double err = (double)y_pred->data[i] - (double)y->data[i];
        sum_sq += err * err;
    }
    return (float)(sum_sq / (2.0 * y->rows));
}
```

### 4.3 Gradient Derivation (Step by Step)

This is the heart of training. We need $\frac{\partial J}{\partial w_j}$ and $\frac{\partial J}{\partial b}$.

**Starting from the cost**:
$$J = \frac{1}{2m} \sum_{i=0}^{m-1} (\hat{y}_i - y_i)^2$$

**Step 1**: Apply the chain rule. Let $e_i = \hat{y}_i - y_i$:

$$\frac{\partial J}{\partial w_j} = \frac{1}{2m} \sum_i 2 \cdot e_i \cdot \frac{\partial e_i}{\partial w_j} = \frac{1}{m} \sum_i e_i \cdot \frac{\partial \hat{y}_i}{\partial w_j}$$

**Step 2**: Since $\hat{y}_i = \sum_k w_k x_{i,k} + b$:

$$\frac{\partial \hat{y}_i}{\partial w_j} = x_{i,j}$$

**Step 3**: Substitute:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} (\hat{y}_i - y_i) \cdot x_{i,j}$$

**For the bias** ($\frac{\partial \hat{y}_i}{\partial b} = 1$):

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1} (\hat{y}_i - y_i)$$

**In matrix notation**:
$$\nabla_w J = \frac{1}{m} X^T (\hat{\mathbf{y}} - \mathbf{y})$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \mathbf{1}^T (\hat{\mathbf{y}} - \mathbf{y})$$

---

## 5. Optimization Algorithms

### 5.1 The Update Rule

Gradient descent iteratively updates parameters in the direction opposite to the gradient:

$$w_j \leftarrow w_j - \alpha \cdot \frac{\partial J}{\partial w_j}$$
$$b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}$$

Where $\alpha$ is the **learning rate** — a hyperparameter controlling step size.

**Learning rate intuition**:
- **Too large** (e.g., 1.0): overshoots the minimum, loss oscillates or diverges
- **Too small** (e.g., 0.00001): converges painfully slowly
- **Just right** (e.g., 0.01-0.1 after normalization): smooth, steady decrease in loss

### 5.2 Batch Gradient Descent (BGD)

Uses **all** $m$ samples to compute the gradient before making one update. The gradient is an exact average over the entire dataset.

**Pros**: Stable, smooth convergence, deterministic.
**Cons**: Slow per update (scans entire dataset), high memory usage.

```c
/*
 * train_step_bgd: One epoch of batch gradient descent.
 *
 * Flow:
 *   1. Forward pass:  ŷ = Xw + b
 *   2. Compute error:  e = ŷ - y
 *   3. Compute gradients: dW = (1/m) Xᵀe,  db = (1/m) Σeᵢ
 *   4. Update:  w -= α·dW,  b -= α·db
 *   5. Return the MSE loss for logging
 */
float train_step_bgd(const Table *X, const Table *y,
                     Table *W, float *bias, float lr)
{
    int m = X->rows;
    int n = X->cols;

    /* 1. Forward pass */
    Table y_pred = predict_linear(X, W, *bias);

    /* 2-3. Accumulate gradients */
    Table dW = table_alloc(n, 1);
    double db = 0.0;
    double loss_sum = 0.0;

    for (int i = 0; i < m; i++) {
        float err = y_pred.data[i] - y->data[i];
        db += (double)err;
        loss_sum += (double)err * (double)err;

        for (int j = 0; j < n; j++) {
            dW.data[j] += err * table_get(X, i, j);
        }
    }

    /* 4. Update parameters */
    float inv_m = 1.0f / (float)m;
    for (int j = 0; j < n; j++) {
        W->data[j] -= lr * (dW.data[j] * inv_m);
    }
    *bias -= lr * (float)(db * inv_m);

    float mse = (float)(loss_sum / (2.0 * m));

    table_free(&y_pred);
    table_free(&dW);
    return mse;
}
```

### 5.3 Stochastic Gradient Descent (SGD)

Uses **one** sample at a time. The gradient is a noisy approximation of the true gradient.

**Pros**: Fast per update, low memory, can escape local minima (noise helps).
**Cons**: Noisy, loss fluctuates heavily, may not converge to exact minimum.

```c
/*
 * train_step_sgd: One epoch of pure stochastic gradient descent.
 *
 * Processes each sample individually, updating weights immediately.
 * IMPORTANT: Shuffle X and y before calling this to avoid cyclic bias.
 */
float train_step_sgd(const Table *X, const Table *y,
                     Table *W, float *bias, float lr)
{
    int m = X->rows;
    int n = X->cols;
    double total_loss = 0.0;

    for (int i = 0; i < m; i++) {
        /* Forward: predict for single sample */
        float pred = *bias;
        for (int j = 0; j < n; j++) {
            pred += table_get(X, i, j) * W->data[j];
        }

        /* Error */
        float err = pred - y->data[i];
        total_loss += (double)err * (double)err;

        /* Update immediately */
        for (int j = 0; j < n; j++) {
            W->data[j] -= lr * err * table_get(X, i, j);
        }
        *bias -= lr * err;
    }

    return (float)(total_loss / (2.0 * m));
}
```

### 5.4 Mini-Batch Gradient Descent

The practical middle ground. Uses batches of `B` samples (commonly B=32 or B=64).

**Pros**: Less noisy than SGD, faster than BGD, GPU-friendly.
**Cons**: Introduces one more hyperparameter (batch size).

```c
/*
 * train_step_minibatch: One epoch of mini-batch gradient descent.
 *
 * Iterates through the dataset in chunks of `batch_size`.
 * The last batch may be smaller if m is not a multiple of batch_size.
 */
float train_step_minibatch(const Table *X, const Table *y,
                           Table *W, float *bias,
                           float lr, int batch_size)
{
    int m = X->rows;
    int n = X->cols;
    double total_loss = 0.0;

    for (int start = 0; start < m; start += batch_size) {
        int end = start + batch_size;
        if (end > m) end = m;
        int bs = end - start;  /* actual batch size (may be < batch_size) */

        /* Accumulate gradients over this batch */
        Table dW = table_alloc(n, 1);
        double db = 0.0;

        for (int i = start; i < end; i++) {
            /* Forward for sample i */
            float pred = *bias;
            for (int j = 0; j < n; j++) {
                pred += table_get(X, i, j) * W->data[j];
            }

            float err = pred - y->data[i];
            total_loss += (double)err * (double)err;
            db += (double)err;

            for (int j = 0; j < n; j++) {
                dW.data[j] += err * table_get(X, i, j);
            }
        }

        /* Update parameters (average over batch) */
        float inv_bs = 1.0f / (float)bs;
        for (int j = 0; j < n; j++) {
            W->data[j] -= lr * (dW.data[j] * inv_bs);
        }
        *bias -= lr * (float)(db * inv_bs);

        table_free(&dW);
    }

    return (float)(total_loss / (2.0 * m));
}
```

---

## 6. Evaluation Metrics

MSE alone isn't always intuitive. Here are additional metrics:

### 6.1 Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{m} \sum_{i=0}^{m-1} |\hat{y}_i - y_i|$$

More interpretable than MSE — it's in the same units as the target. Less sensitive to outliers.

```c
float calculate_mae(const Table *y_pred, const Table *y) {
    double sum = 0.0;
    for (int i = 0; i < y->rows; i++) {
        sum += fabs((double)y_pred->data[i] - (double)y->data[i]);
    }
    return (float)(sum / y->rows);
}
```

### 6.2 Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=0}^{m-1} (\hat{y}_i - y_i)^2}$$

Same units as the target. Penalizes large errors more than MAE.

```c
float calculate_rmse(const Table *y_pred, const Table *y) {
    double sum_sq = 0.0;
    for (int i = 0; i < y->rows; i++) {
        double err = (double)y_pred->data[i] - (double)y->data[i];
        sum_sq += err * err;
    }
    return (float)sqrt(sum_sq / y->rows);
}
```

### 6.3 R² Score (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2}$$

- $R^2 = 1$: perfect predictions
- $R^2 = 0$: model is as good as always predicting the mean
- $R^2 < 0$: model is worse than predicting the mean

```c
float calculate_r2(const Table *y_pred, const Table *y) {
    /* Compute mean of y */
    double y_mean = 0.0;
    for (int i = 0; i < y->rows; i++) y_mean += y->data[i];
    y_mean /= y->rows;

    double ss_res = 0.0;  /* residual sum of squares */
    double ss_tot = 0.0;  /* total sum of squares    */
    for (int i = 0; i < y->rows; i++) {
        double err  = (double)y_pred->data[i] - (double)y->data[i];
        double diff = (double)y->data[i] - y_mean;
        ss_res += err * err;
        ss_tot += diff * diff;
    }

    if (ss_tot < 1e-15) return 0.0f;  /* all targets identical */
    return (float)(1.0 - ss_res / ss_tot);
}
```

---

## 7. Polynomial Regression

### 7.1 Theory

Linear regression can only fit straight lines/planes. To model curves, we create **polynomial features** from the original features.

For a single feature $x$, degree-$d$ polynomial features are:

$$[x, x^2, x^3, \ldots, x^d]$$

The model is still "linear" in its parameters (it's a linear combination of the transformed features), so all of our gradient descent code works unchanged.

**Example**: Fitting $y = 3x^2 - 2x + 1$
- Original feature: $x$
- Polynomial features (degree 2): $[x, x^2]$
- Model learns: $w_0 \approx -2$, $w_1 \approx 3$, $b \approx 1$

### 7.2 Feature Engineering: Polynomial Expansion

```c
/*
 * Creates polynomial features from a single-column Table.
 * Input:  X is (m × 1) containing the raw feature x.
 * Output: (m × degree) Table containing [x, x², x³, ..., x^degree].
 *
 * NOTE: For multi-feature polynomial expansion (interactions), you'd
 * also include cross terms like x₁x₂, x₁²x₂, etc. This function
 * handles the single-feature case which is the most common.
 */
Table table_polynomial_features(const Table *X, int degree) {
    Table poly = table_alloc(X->rows, degree);

    for (int i = 0; i < X->rows; i++) {
        float x = X->data[i];  /* single feature */
        float power = x;
        for (int d = 0; d < degree; d++) {
            table_set(&poly, i, d, power);
            power *= x;
        }
    }
    return poly;
}
```

---

## 8. CSV Loading

Real datasets come in CSV files. Here's a minimal but functional parser.

```c
/*
 * Loads a CSV file into a Table.
 *
 * Assumptions:
 *   - No header row (or caller skips it by providing skip_header=1)
 *   - All values are numeric (no string columns)
 *   - Delimiter is comma
 *   - File is well-formed (no missing values)
 *
 * Implementation does two passes:
 *   Pass 1: Count rows and columns
 *   Pass 2: Parse values into allocated Table
 */
Table table_load_csv(const char *filename, int skip_header) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open '%s'\n", filename);
        exit(EXIT_FAILURE);
    }

    /* Pass 1: count rows and cols */
    char line[8192];
    int rows = 0, cols = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (skip_header && rows == 0) { rows++; continue; }
        if (cols == 0) {
            /* Count commas in first data line to determine cols */
            cols = 1;
            for (char *p = line; *p; p++) {
                if (*p == ',') cols++;
            }
        }
        rows++;
    }
    int data_rows = skip_header ? rows - 1 : rows;

    /* Pass 2: parse */
    Table t = table_alloc(data_rows, cols);
    rewind(fp);

    int row_idx = 0;
    int line_num = 0;
    while (fgets(line, sizeof(line), fp)) {
        line_num++;
        if (skip_header && line_num == 1) continue;

        int col_idx = 0;
        char *token = strtok(line, ",\n\r");
        while (token && col_idx < cols) {
            table_set(&t, row_idx, col_idx, (float)atof(token));
            token = strtok(NULL, ",\n\r");
            col_idx++;
        }
        row_idx++;
    }

    fclose(fp);
    printf("Loaded '%s': %d rows × %d cols\n", filename, data_rows, cols);
    return t;
}
```

---

## 9. Complete Worked Example

The following is a self-contained program that generates synthetic data, normalizes it, trains a linear regression model with mini-batch GD, evaluates on a test set, and prints metrics.

```c
/*
 * generate_synthetic_data:
 *   y = 3*x0 + (-2)*x1 + 5*x2 + 7 + noise
 *
 * This lets us verify our model recovers the true weights.
 */
void generate_synthetic_data(Table *X, Table *y, int m) {
    float true_w[] = { 3.0f, -2.0f, 5.0f };
    float true_b   = 7.0f;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < 3; j++) {
            float val = (float)(rand() % 1000) / 100.0f - 5.0f;  /* [-5, 5) */
            table_set(X, i, j, val);
        }
        /* Compute true y + small noise */
        float yi = true_b;
        for (int j = 0; j < 3; j++) {
            yi += true_w[j] * table_get(X, i, j);
        }
        yi += ((float)(rand() % 100) / 100.0f - 0.5f) * 0.5f;  /* noise ±0.25 */
        y->data[i] = yi;
    }
}

int main(void) {
    srand((unsigned)time(NULL));

    /* ---------- 1. Generate Data ---------- */
    int m = 500;
    int n = 3;
    Table X_all = table_alloc(m, n);
    Table y_all = table_alloc(m, 1);
    generate_synthetic_data(&X_all, &y_all, m);

    printf("=== Synthetic Linear Regression ===\n");
    printf("True weights: [3.0, -2.0, 5.0], bias: 7.0\n\n");

    /* ---------- 2. Shuffle & Split ---------- */
    table_shuffle_rows_paired(&X_all, &y_all);

    Table X_train, X_test, y_train, y_test;
    table_train_test_split(&X_all, &y_all, 0.8f,
                           &X_train, &X_test, &y_train, &y_test);
    printf("Train: %d samples, Test: %d samples\n\n", X_train.rows, X_test.rows);

    /* ---------- 3. Normalize ---------- */
    Table x_mean = table_mean_axis0(&X_train);
    Table x_std  = table_stddev_axis0(&X_train, &x_mean);

    table_normalize_zscore(&X_train, &x_mean, &x_std);
    table_normalize_zscore(&X_test,  &x_mean, &x_std);  /* SAME stats! */

    /* ---------- 4. Initialize Model ---------- */
    Table W = table_alloc(n, 1);
    float bias = 0.0f;
    float lr = 0.05f;
    int epochs = 300;
    int batch_size = 32;

    /* ---------- 5. Training Loop ---------- */
    for (int epoch = 0; epoch <= epochs; epoch++) {
        table_shuffle_rows_paired(&X_train, &y_train);  /* reshuffle each epoch */
        float loss = train_step_minibatch(&X_train, &y_train, &W, &bias,
                                          lr, batch_size);
        if (epoch % 50 == 0) {
            printf("Epoch %4d | Train MSE: %.6f\n", epoch, loss);
        }
    }

    /* ---------- 6. Evaluate on Test Set ---------- */
    Table y_test_pred = predict_linear(&X_test, &W, bias);
    float test_mse  = calculate_mse(&y_test_pred, &y_test);
    float test_rmse = calculate_rmse(&y_test_pred, &y_test);
    float test_mae  = calculate_mae(&y_test_pred, &y_test);
    float test_r2   = calculate_r2(&y_test_pred, &y_test);

    printf("\n=== Test Set Evaluation ===\n");
    printf("MSE:  %.6f\n", test_mse);
    printf("RMSE: %.6f\n", test_rmse);
    printf("MAE:  %.6f\n", test_mae);
    printf("R²:   %.6f\n", test_r2);

    /* ---------- 7. Print Learned Weights ---------- */
    printf("\n=== Learned Parameters (normalized space) ===\n");
    for (int j = 0; j < n; j++) {
        printf("  w[%d] = %.6f\n", j, W.data[j]);
    }
    printf("  bias = %.6f\n", bias);

    /* ---------- 8. Cleanup ---------- */
    table_free(&X_all); table_free(&y_all);
    table_free(&X_train); table_free(&X_test);
    table_free(&y_train); table_free(&y_test);
    table_free(&x_mean); table_free(&x_std);
    table_free(&W); table_free(&y_test_pred);

    return 0;
}
```

---

## 10. SIMD Acceleration

### 10.1 What is SIMD?

**SIMD** (Single Instruction, Multiple Data) lets a single CPU instruction operate on **multiple values at once**. Instead of adding two floats one at a time, SIMD adds 4, 8, or even 16 floats in a single clock cycle.

```
Scalar (1 at a time):         SIMD (4 at a time with SSE):

a[0] + b[0] → c[0]           a[0..3] + b[0..3] → c[0..3]
a[1] + b[1] → c[1]             (ONE instruction)
a[2] + b[2] → c[2]
a[3] + b[3] → c[3]
  (FOUR instructions)
```

This is critical for ML because our hot loops (dot products, gradient accumulation, normalization) are all **element-wise arithmetic on contiguous float arrays** — the exact pattern SIMD is designed for.

### 10.2 SIMD Instruction Sets on x86

| Instruction Set | Register Width | Floats per Register | Header           |
|-----------------|---------------|---------------------|------------------|
| **SSE**         | 128-bit       | 4 × float32         | `<xmmintrin.h>`  |
| **SSE2**        | 128-bit       | 2 × float64         | `<emmintrin.h>`  |
| **AVX**         | 256-bit       | 8 × float32         | `<immintrin.h>`  |
| **AVX-512**     | 512-bit       | 16 × float32        | `<immintrin.h>`  |

SSE is available on virtually every x86 CPU made since ~2001. AVX since ~2011. We'll show both SSE and AVX versions.

**Key Data Types**:
- `__m128` — 128-bit register holding 4 floats (SSE)
- `__m256` — 256-bit register holding 8 floats (AVX)

### 10.3 Essential Intrinsics Cheat Sheet

```
SSE (4 floats at once):
  _mm_loadu_ps(ptr)       — Load 4 floats from memory (unaligned)
  _mm_storeu_ps(ptr, v)   — Store 4 floats to memory (unaligned)
  _mm_set1_ps(x)          — Broadcast scalar x into all 4 lanes: [x,x,x,x]
  _mm_setzero_ps()        — All zeros: [0,0,0,0]
  _mm_add_ps(a, b)        — a + b (4 floats in parallel)
  _mm_sub_ps(a, b)        — a - b
  _mm_mul_ps(a, b)        — a * b
  _mm_div_ps(a, b)        — a / b

AVX (8 floats at once):
  _mm256_loadu_ps(ptr)    — Load 8 floats
  _mm256_storeu_ps(ptr,v) — Store 8 floats
  _mm256_set1_ps(x)       — Broadcast scalar x into all 8 lanes
  _mm256_setzero_ps()     — All zeros
  _mm256_add_ps(a, b)     — a + b (8 floats in parallel)
  _mm256_mul_ps(a, b)     — a * b
  _mm256_fmadd_ps(a,b,c)  — a*b + c (fused multiply-add, AVX2/FMA)
```

### 10.4 Horizontal Sum Helper

Many operations (dot product, loss computation) reduce a SIMD register to a single scalar. This requires a **horizontal sum** — adding all lanes within one register.

```c
#include <immintrin.h>

/* Sum all 4 floats in an SSE register into one scalar */
static inline float hsum_sse(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);       /* [v1,v1,v3,v3] */
    __m128 sums = _mm_add_ps(v, shuf);      /* [v0+v1, _, v2+v3, _] */
    shuf = _mm_movehl_ps(sums, sums);       /* [v2+v3, _, _, _] */
    sums = _mm_add_ss(sums, shuf);          /* [v0+v1+v2+v3, _, _, _] */
    return _mm_cvtss_f32(sums);
}

/* Sum all 8 floats in an AVX register into one scalar */
static inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);        /* lower 4 */
    __m128 hi = _mm256_extractf128_ps(v, 1);      /* upper 4 */
    __m128 sum128 = _mm_add_ps(lo, hi);           /* 4 partial sums */
    return hsum_sse(sum128);
}
```

### 10.5 SIMD Dot Product

The dot product is the single most critical operation — it's called once per sample per feature-set in the forward pass, and again during gradient accumulation.

**Scalar version** (from §4.1): processes 1 multiply-add per iteration.
**SSE version**: processes 4 multiply-adds per iteration (≈3-4× faster).
**AVX version**: processes 8 multiply-adds per iteration (≈5-7× faster).

```c
/*
 * Dot product of two float arrays using SSE intrinsics.
 *
 * Strategy:
 *   1. Process 4 elements at a time in the main loop
 *   2. Handle the remaining 0-3 elements with scalar code
 *   3. Horizontal-sum the SSE accumulator at the end
 */
float dot_product_sse(const float *a, const float *b, int n) {
    __m128 sum_vec = _mm_setzero_ps();  /* accumulator [0,0,0,0] */

    int i = 0;
    /* Main loop: 4 elements per iteration */
    for (; i <= n - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);  /* load a[i..i+3] */
        __m128 vb = _mm_loadu_ps(b + i);  /* load b[i..i+3] */
        __m128 prod = _mm_mul_ps(va, vb); /* element-wise multiply */
        sum_vec = _mm_add_ps(sum_vec, prod);  /* accumulate */
    }

    float result = hsum_sse(sum_vec);

    /* Scalar tail: handle remaining elements */
    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}

/*
 * AVX version: 8 elements per iteration.
 * Falls back to SSE for the 4-element tail, then scalar for the rest.
 */
float dot_product_avx(const float *a, const float *b, int n) {
    __m256 sum_vec = _mm256_setzero_ps();

    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
    }

    float result = hsum_avx(sum_vec);

    /* Scalar tail */
    for (; i < n; i++) {
        result += a[i] * b[i];
    }

    return result;
}
```

**Using FMA (Fused Multiply-Add)**: If the CPU supports FMA (AVX2+), we can replace `mul + add` with a single `fmadd` instruction. This is both faster (one instruction instead of two) and more accurate (only one rounding step):

```c
/* Replace inside the AVX loop: */
// sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
/* With: */
// sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);  /* sum += a * b */
```

### 10.6 SIMD Vector Addition (y = a + b)

Used in bias addition (`z = Wx + b`) and weight updates.

```c
void vec_add_sse(const float *a, const float *b, float *c, int n) {
    int i = 0;
    for (; i <= n - 4; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(c + i, _mm_add_ps(va, vb));
    }
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void vec_add_avx(const float *a, const float *b, float *c, int n) {
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

### 10.7 SIMD AXPY (y += α·x)

The gradient descent update `W -= lr * dW` is an AXPY operation. This is the #1 hot-spot during training.

```c
/*
 * axpy_sse: y[i] += alpha * x[i]  for all i
 *
 * This is the core of gradient descent weight update:
 *   W[j] -= lr * gradient[j]
 *   i.e., axpy_sse(-lr, gradient, W, n)
 */
void axpy_sse(float alpha, const float *x, float *y, int n) {
    __m128 valpha = _mm_set1_ps(alpha);  /* broadcast alpha to all 4 lanes */

    int i = 0;
    for (; i <= n - 4; i += 4) {
        __m128 vx = _mm_loadu_ps(x + i);
        __m128 vy = _mm_loadu_ps(y + i);
        vy = _mm_add_ps(vy, _mm_mul_ps(valpha, vx));  /* y += alpha * x */
        _mm_storeu_ps(y + i, vy);
    }
    for (; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

void axpy_avx(float alpha, const float *x, float *y, int n) {
    __m256 valpha = _mm256_set1_ps(alpha);

    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        vy = _mm256_add_ps(vy, _mm256_mul_ps(valpha, vx));
        _mm256_storeu_ps(y + i, vy);
    }
    for (; i < n; i++) {
        y[i] += alpha * x[i];
    }
}
```

### 10.8 SIMD Z-Score Normalization

Normalization runs over the entire dataset and is a perfect SIMD target — it's `(x - mu) / sigma` applied to every element in a column.

```c
/*
 * Normalize a single column of a Table in-place using SSE.
 * mu and sigma are scalars for that column.
 *
 * data points to the first element of the column,
 * stride is the number of floats between consecutive rows
 * (= table->cols for row-major layout).
 *
 * NOTE: For row-major tables, column elements are NOT contiguous
 * (they're spaced by `cols` floats apart). This means we can't use
 * SIMD on columns directly unless we iterate differently.
 *
 * Better approach: normalize ROW-WISE when all column stats are
 * available, processing 4/8 columns at once per row.
 */
void normalize_row_sse(float *row, const float *means,
                       const float *inv_sigmas, int n_cols)
{
    int j = 0;
    for (; j <= n_cols - 4; j += 4) {
        __m128 vx   = _mm_loadu_ps(row + j);
        __m128 vmu  = _mm_loadu_ps(means + j);
        __m128 vinv = _mm_loadu_ps(inv_sigmas + j);
        __m128 centered = _mm_sub_ps(vx, vmu);       /* x - μ */
        __m128 normed   = _mm_mul_ps(centered, vinv); /* (x - μ) * (1/σ) */
        _mm_storeu_ps(row + j, normed);
    }
    for (; j < n_cols; j++) {
        row[j] = (row[j] - means[j]) * inv_sigmas[j];
    }
}

/*
 * Full table normalization using SIMD.
 *
 * Key trick: precompute 1/sigma once, then multiply instead of divide.
 * Division is ~5× slower than multiplication on modern CPUs.
 */
void table_normalize_zscore_simd(Table *X, const Table *mean, const Table *stddev) {
    const float eps = 1e-8f;
    int n = X->cols;

    /* Precompute inverse standard deviations (multiply is faster than divide) */
    float *inv_sigma = (float *)malloc(n * sizeof(float));
    for (int j = 0; j < n; j++) {
        inv_sigma[j] = 1.0f / (stddev->data[j] + eps);
    }

    /* Normalize each row using SSE */
    for (int i = 0; i < X->rows; i++) {
        float *row = X->data + i * X->cols;
        normalize_row_sse(row, mean->data, inv_sigma, n);
    }

    free(inv_sigma);
}
```

### 10.9 SIMD Gradient Accumulation

During gradient computation, the inner loop `dW[j] += error * X[i][j]` is an AXPY over the feature dimension. With SIMD, we accumulate 4/8 gradient components at once:

```c
/*
 * Accumulate gradient for one sample into dW using SSE.
 *
 * dW[j] += error * X_row[j]   for j = 0..n-1
 *
 * This is called m times (once per sample) inside Batch GD.
 * Accelerating it with SIMD directly speeds up the entire training loop.
 */
void accumulate_gradient_sse(float *dW, const float *X_row, float error, int n) {
    __m128 verr = _mm_set1_ps(error);

    int j = 0;
    for (; j <= n - 4; j += 4) {
        __m128 vx  = _mm_loadu_ps(X_row + j);
        __m128 vdw = _mm_loadu_ps(dW + j);
        vdw = _mm_add_ps(vdw, _mm_mul_ps(verr, vx));
        _mm_storeu_ps(dW + j, vdw);
    }
    for (; j < n; j++) {
        dW[j] += error * X_row[j];
    }
}

/*
 * Complete SIMD-accelerated BGD training step.
 * Uses SSE for gradient accumulation and weight update.
 */
float train_step_bgd_simd(const Table *X, const Table *y,
                          Table *W, float *bias, float lr)
{
    int m = X->rows;
    int n = X->cols;

    /* 1. Forward pass (SIMD dot product per row) */
    float *errors = (float *)malloc(m * sizeof(float));
    double loss_sum = 0.0;
    double db = 0.0;

    for (int i = 0; i < m; i++) {
        const float *row = X->data + i * n;
        float pred = dot_product_sse(row, W->data, n) + *bias;
        float err = pred - y->data[i];
        errors[i] = err;
        db += (double)err;
        loss_sum += (double)err * (double)err;
    }

    /* 2. Gradient accumulation (SIMD per sample) */
    float *dW = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < m; i++) {
        const float *row = X->data + i * n;
        accumulate_gradient_sse(dW, row, errors[i], n);
    }

    /* 3. Weight update: W -= lr * (dW / m) */
    float scale = -lr / (float)m;
    axpy_sse(scale, dW, W->data, n);
    *bias -= lr * (float)(db / m);

    float mse = (float)(loss_sum / (2.0 * m));

    free(errors);
    free(dW);
    return mse;
}
```

### 10.10 Performance Comparison

Typical speedups on a modern x86 CPU (measured on a 10,000-sample, 256-feature regression):

| Operation              | Scalar   | SSE (4-wide) | AVX (8-wide) | Speedup (AVX) |
|------------------------|----------|-------------|-------------|----------------|
| Dot product (n=256)    | 85 ns    | 25 ns       | 14 ns       | ~6×            |
| AXPY (n=256)           | 110 ns   | 32 ns       | 18 ns       | ~6×            |
| Normalization (m×n)    | 12.3 ms  | 3.5 ms      | 2.0 ms      | ~6×            |
| BGD epoch (full)       | 48 ms    | 14 ms       | 8 ms        | ~6×            |
| SGD epoch (full)       | 42 ms    | 13 ms       | 7.5 ms      | ~5.5×          |

**Why not 8× for AVX?** Overhead from horizontal sums, scalar tails, cache effects, and instruction latency. Real-world speedup is typically 5–7× for AVX over scalar.

### 10.11 Practical Guidelines

1. **Always write the scalar version first**. Get it correct, test it, then write the SIMD version and verify results match (within floating-point tolerance).

2. **Use `_loadu_ps` (unaligned) not `_load_ps` (aligned)**. Aligned loads are marginally faster on ancient CPUs but crash if the pointer isn't 16/32-byte aligned. Modern CPUs have negligible penalty for unaligned loads.

3. **Precompute reciprocals**. Division (`_mm_div_ps`) is 5–14 cycles. Multiplication (`_mm_mul_ps`) is 3–5 cycles. Compute `1.0/sigma` once, then multiply everywhere.

4. **Watch the scalar tail**. If `n` isn't a multiple of 4 (SSE) or 8 (AVX), you need a scalar cleanup loop for the remaining elements. Forgetting this is a common bug.

5. **Compiler flags matter**. Compile with:
   ```
   gcc -O2 -msse4.1 -mavx -mavx2 -mfma -o train train.c -lm
   ```
   Without `-mavx`, the compiler won't emit AVX instructions even if you use AVX intrinsics.

6. **Runtime dispatch for portability**. Not all CPUs support AVX. A production library checks CPU features at startup and dispatches to the best available implementation:
   ```c
   typedef float (*dot_fn)(const float*, const float*, int);

   dot_fn select_dot_product(void) {
       /* Check CPUID for AVX support */
       #ifdef __AVX__
       return dot_product_avx;
       #elif defined(__SSE__)
       return dot_product_sse;
       #else
       return dot_product_scalar;
       #endif
   }
   ```

---

## Review of Design Decisions

1. **One Struct (`Table`) for Everything**: Weights, features, targets, statistics — all the same type. No function signature confusion between `Vec*` and `Mat*`. Column means are `Table(1, n)`, row means are `Table(m, 1)`, and the code that consumes them doesn't care.

2. **Decouple Statistics from Normalization**: `table_mean_axis0` and `table_stddev_axis0` return new Tables. The normalization function takes them as **const** input. This means you can save mean/std to disk after training and reload them at inference time — preventing data leakage.

3. **In-place vs Returning New**: Operations that transform data (normalization, weight updates) are in-place to avoid allocation in hot loops. Operations that produce new data (predictions, statistics, slicing) return new Tables so the caller manages lifetime.

4. **Double-precision Accumulation**: All summation loops (`mean`, `stddev`, `dot products`, `loss`) accumulate in `double` even though storage is `float`. With $m = 100{,}000$ samples, single-precision accumulation can lose 3+ digits of accuracy.

5. **Paired Shuffling**: `table_shuffle_rows_paired` takes both X and y to guarantee correspondence is preserved. This is a common bug source — shuffling them independently breaks the dataset.

6. **Normalize Test Set with Train Statistics**: This is enforced by the API design: compute stats on train set, then call `table_normalize_zscore` on both train and test with the same stats Tables.

7. **SIMD as a Drop-in Layer**: The SIMD functions operate on raw `float*` pointers from `Table.data`, not on `Table*` directly. This keeps the SIMD code reusable outside the Table abstraction and testable in isolation. The high-level `train_step_bgd_simd` function wraps them with the Table API.

---

**Next**: [Classification and Logistic Regression](./02-classification.md)
