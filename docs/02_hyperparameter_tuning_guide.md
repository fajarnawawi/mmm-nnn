# Hyperparameter Tuning Guide for NNN Marketing Mix Modeling

## Overview

This guide provides a systematic approach to tuning the NNN model's hyperparameters for optimal performance. The paper identifies L1 regularization and loss balancing as the primary tuning parameters, but we also cover architectural and optimization hyperparameters.

---

## 1. Hyperparameter Categories

### 1.1 Critical Hyperparameters (Primary Focus)

These have the **largest impact** on model performance according to the paper:

| Parameter | Description | Default | Range to Search |
|-----------|-------------|---------|-----------------|
| `l1_lambda` | L1 regularization strength | 1e-2 | [1e-4, 1e-1] |
| `alpha` | Sales vs. Search loss balance | 0.9 | [0.7, 0.99] |

### 1.2 Architecture Hyperparameters

| Parameter | Description | Default | Range to Search |
|-----------|-------------|---------|-----------------|
| `n_transformer_layers` | Number of transformer blocks | 2 | [1, 2, 3, 4] |
| `transformer_ff_size` | Feed-forward layer size | 256 | [128, 256, 512] |
| `embed_dim` | Embedding dimension | 64 | [32, 64, 128, 256] |
| `sales_head_layers` | MLP depth in sales head | 4 | [2, 4, 6] |
| `sales_head_size` | Hidden layer size in sales head | 64 | [32, 64, 128] |

### 1.3 Optimization Hyperparameters

| Parameter | Description | Default | Range to Search |
|-----------|-------------|---------|-----------------|
| `learning_rate` | Adam optimizer learning rate | 1e-3 | [1e-4, 1e-2] |
| `train_steps` | Number of training iterations | 1000 | [500, 1000, 5000] |
| `lookback_window` | Temporal attention window | 52 | [26, 52, 104] |

---

## 2. Tuning Strategy

### 2.1 Recommended Tuning Order

**Stage 1: Critical Parameters (Primary)**
1. `l1_lambda` - Regularization strength
2. `alpha` - Loss balance

**Stage 2: Architecture (Secondary)**
3. `embed_dim` - Embedding size
4. `n_transformer_layers` - Model depth
5. `transformer_ff_size` - Model width

**Stage 3: Fine-tuning (Tertiary)**
6. `learning_rate` - Optimization speed
7. `train_steps` - Training duration
8. Sales head architecture

---

## 3. Grid Search Implementation

### 3.1 Stage 1: Critical Parameters

**Grid Search for L1 Lambda and Alpha:**

```python
import itertools
from typing import Dict, List, Tuple

def grid_search_critical(X_train, X_val, model_class,
                         l1_lambdas=[1e-4, 1e-3, 1e-2, 1e-1],
                         alphas=[0.7, 0.8, 0.9, 0.95, 0.99]):
    """
    Grid search over critical hyperparameters.

    Args:
        X_train: Training data tensor
        X_val: Validation data tensor
        model_class: NNN model class
        l1_lambdas: List of L1 regularization values to try
        alphas: List of loss balance values to try

    Returns:
        best_params: Dictionary with best hyperparameters
        results: DataFrame with all results
    """

    results = []

    # Grid search
    for l1_lambda, alpha in itertools.product(l1_lambdas, alphas):
        print(f"\nTrying l1_lambda={l1_lambda}, alpha={alpha}")

        # Initialize model with these hyperparameters
        key = PRNGKey(42)
        model = model_class(index_mapping=CHANNEL_MAP_FROZEN)
        params = model.init(key, X_train)

        # Train
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        train_losses = []
        for step in range(1000):
            params, opt_state, loss, _, _ = train_step(
                params, X_train, opt_state
            )
            train_losses.append(float(loss))

        # Evaluate on validation set
        sales_pred, search_pred = model.apply(params, X_val)

        # Calculate validation metrics
        sales_true = X_val[:, :, 0, 0]
        val_sales_mse = float(jnp.mean((sales_true - sales_pred)**2))
        val_sales_mae = float(jnp.mean(jnp.abs(sales_true - sales_pred)))
        val_sales_mape = float(jnp.mean(jnp.abs((sales_true - sales_pred) / (sales_true + 1e-8))))

        # Store results
        results.append({
            'l1_lambda': l1_lambda,
            'alpha': alpha,
            'val_sales_mse': val_sales_mse,
            'val_sales_mae': val_sales_mae,
            'val_sales_mape': val_sales_mape,
            'train_loss_final': train_losses[-1]
        })

        print(f"  Val MSE: {val_sales_mse:.4f}, Val MAE: {val_sales_mae:.4f}, Val MAPE: {val_sales_mape:.4f}")

    # Find best parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['val_sales_mse'].idxmin()
    best_params = results_df.iloc[best_idx].to_dict()

    print("\n" + "="*60)
    print("BEST PARAMETERS:")
    print(f"  l1_lambda: {best_params['l1_lambda']}")
    print(f"  alpha: {best_params['alpha']}")
    print(f"  Val MSE: {best_params['val_sales_mse']:.4f}")
    print("="*60)

    return best_params, results_df

# Run grid search
best_params, grid_results = grid_search_critical(X_train, X_val, NNN)
```

**Visualize Results:**

```python
import plotly.graph_objects as go

# Heatmap of validation MSE
pivot = grid_results.pivot(index='l1_lambda', columns='alpha', values='val_sales_mse')

fig = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns,
    y=pivot.index,
    colorscale='Viridis',
    text=pivot.values,
    texttemplate='%{text:.2f}',
    textfont={"size": 10}
))

fig.update_layout(
    title='Validation MSE by L1 Lambda and Alpha',
    xaxis_title='Alpha (Sales Loss Weight)',
    yaxis_title='L1 Lambda (Regularization)',
    yaxis_type='log'
)
fig.show()
```

---

### 3.2 Stage 2: Architecture Search

**Once you have optimal L1/Alpha, search architecture:**

```python
def architecture_search(X_train, X_val, best_l1, best_alpha):
    """
    Search over architecture hyperparameters.
    """

    architectures = [
        {'embed_dim': 32, 'n_layers': 2, 'ff_size': 128},
        {'embed_dim': 64, 'n_layers': 2, 'ff_size': 256},
        {'embed_dim': 128, 'n_layers': 2, 'ff_size': 256},
        {'embed_dim': 64, 'n_layers': 3, 'ff_size': 256},
        {'embed_dim': 64, 'n_layers': 4, 'ff_size': 512},
    ]

    results = []

    for arch in architectures:
        print(f"\nTrying architecture: {arch}")

        # Note: This requires modifying the NNN class to accept these params
        # Or creating different model instances

        # Train and evaluate
        # ... (similar to above)

        results.append({
            **arch,
            'val_mse': val_mse,
            'train_time': train_time,
            'n_parameters': count_parameters(model)
        })

    return pd.DataFrame(results)
```

---

## 4. Train/Validation/Test Split Strategy

### 4.1 Time-Based Split (Recommended)

**For time series data, use temporal split:**

```python
def create_time_splits(X_tensor, train_frac=0.7, val_frac=0.15):
    """
    Split data temporally for proper validation.

    Args:
        X_tensor: Input tensor (G, T, C, D)
        train_frac: Fraction for training (e.g., 0.7)
        val_frac: Fraction for validation (e.g., 0.15)

    Returns:
        X_train, X_val, X_test
    """
    G, T, C, D = X_tensor.shape

    train_end = int(T * train_frac)
    val_end = int(T * (train_frac + val_frac))

    X_train = X_tensor[:, :train_end, :, :]
    X_val = X_tensor[:, train_end:val_end, :, :]
    X_test = X_tensor[:, val_end:, :, :]

    print(f"Train: weeks 0-{train_end} ({train_end} weeks)")
    print(f"Val: weeks {train_end}-{val_end} ({val_end - train_end} weeks)")
    print(f"Test: weeks {val_end}-{T} ({T - val_end} weeks)")

    return X_train, X_val, X_test

# Example: 70% train, 15% val, 15% test
X_train, X_val, X_test = create_time_splits(X_tensor, 0.7, 0.15)
```

**Important:** Never shuffle time series data!

---

### 4.2 Rolling Window Cross-Validation (Advanced)

**For more robust validation:**

```python
def rolling_window_cv(X_tensor, n_splits=5, test_size=26):
    """
    Perform rolling window cross-validation.

    Args:
        X_tensor: Input tensor (G, T, C, D)
        n_splits: Number of CV folds
        test_size: Size of test window (e.g., 26 weeks)
    """
    G, T, C, D = X_tensor.shape

    results = []

    for i in range(n_splits):
        # Define train/test split for this fold
        test_end = T - (i * test_size)
        test_start = test_end - test_size

        if test_start < test_size * 2:  # Need minimum training data
            break

        X_train_fold = X_tensor[:, :test_start, :, :]
        X_test_fold = X_tensor[:, test_start:test_end, :, :]

        print(f"\nFold {i+1}: Train weeks 0-{test_start}, Test weeks {test_start}-{test_end}")

        # Train model
        # ... (training code)

        # Evaluate
        val_mse = evaluate_model(model, params, X_test_fold)
        results.append(val_mse)

    return {
        'mean_mse': np.mean(results),
        'std_mse': np.std(results),
        'all_mses': results
    }
```

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics

**Sales Prediction Accuracy:**

```python
def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics."""

    metrics = {}

    # Mean Squared Error (MSE)
    metrics['mse'] = float(jnp.mean((y_true - y_pred)**2))

    # Root Mean Squared Error (RMSE)
    metrics['rmse'] = float(jnp.sqrt(metrics['mse']))

    # Mean Absolute Error (MAE)
    metrics['mae'] = float(jnp.mean(jnp.abs(y_true - y_pred)))

    # Mean Absolute Percentage Error (MAPE)
    metrics['mape'] = float(jnp.mean(jnp.abs((y_true - y_pred) / (y_true + 1e-8)))) * 100

    # R-squared
    ss_res = jnp.sum((y_true - y_pred)**2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true))**2)
    metrics['r2'] = float(1 - (ss_res / ss_tot))

    return metrics
```

### 5.2 Attribution Validation Metrics

**If ground truth attribution is available (from experiments):**

```python
def attribution_accuracy(pred_attribution, true_attribution, channel_names):
    """
    Evaluate attribution accuracy.

    Args:
        pred_attribution: Dict of predicted attributions by channel
        true_attribution: Dict of true attributions (from experiments)
        channel_names: List of channel names
    """

    results = []

    for channel in channel_names:
        pred = pred_attribution.get(channel, 0)
        true = true_attribution.get(channel, 0)

        error = abs(pred - true)
        pct_error = (error / (true + 1e-8)) * 100

        results.append({
            'channel': channel,
            'predicted': pred,
            'true': true,
            'error': error,
            'pct_error': pct_error
        })

    return pd.DataFrame(results)
```

---

## 6. Early Stopping

**Prevent overfitting by stopping when validation loss stops improving:**

```python
class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=50, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop

# Usage in training loop
early_stopping = EarlyStopping(patience=50)

for step in range(TRAIN_STEPS):
    # Training step
    params, opt_state, train_loss, _, _ = train_step(...)

    # Evaluate on validation set every N steps
    if step % 10 == 0:
        val_loss = evaluate_on_validation(params, X_val)

        if early_stopping(val_loss):
            print(f"Early stopping at step {step}")
            break
```

---

## 7. Best Practices

### 7.1 General Guidelines

✅ **DO:**
- Start with default parameters, then tune systematically
- Use validation set for all hyperparameter decisions
- Save checkpoints during long training runs
- Track all experiments (use MLflow, Weights & Biases, etc.)
- Compare multiple metrics (MSE, MAE, MAPE, R²)
- Visualize training curves

❌ **DON'T:**
- Tune too many parameters at once
- Use test set for hyperparameter selection
- Cherry-pick best results across different splits
- Ignore training time and model size constraints
- Over-optimize for validation set

---

### 7.2 Computational Efficiency

**Tips for faster tuning:**

```python
# 1. Start with smaller models
embed_dim_candidates = [32, 64]  # Not [32, 64, 128, 256, 512]

# 2. Use fewer training steps initially
quick_train_steps = 200  # For initial search
full_train_steps = 1000  # For final models

# 3. Reduce data size for architecture search
X_train_subset = X_train[:, :52, :, :]  # Use 1 year instead of 3

# 4. Use parallel training for grid search
from joblib import Parallel, delayed

results = Parallel(n_jobs=4)(
    delayed(train_and_evaluate)(hp) for hp in hyperparameter_grid
)
```

---

## 8. Hyperparameter Tuning Checklist

Before finalizing model:

- [ ] **Critical params tuned:** L1 lambda and alpha optimized via grid search
- [ ] **Architecture validated:** Tested at least 3 different architectures
- [ ] **Learning rate schedule:** Tested constant vs. decay
- [ ] **Training duration:** Verified convergence (loss plateaus)
- [ ] **Validation strategy:** Used proper time-based splits
- [ ] **Metrics calculated:** MSE, MAE, MAPE, R² on validation set
- [ ] **Overfitting checked:** Validation loss not increasing while train decreases
- [ ] **Attribution validated:** Checked against holdout experiments (if available)
- [ ] **Final test:** Evaluated on held-out test set (only once!)
- [ ] **Documentation:** Logged all experiments and results

---

## 9. Example: Complete Tuning Workflow

```python
# Step 1: Prepare data splits
X_train, X_val, X_test = create_time_splits(X_tensor, 0.7, 0.15)

# Step 2: Grid search critical parameters
best_params, grid_results = grid_search_critical(
    X_train, X_val, NNN,
    l1_lambdas=[1e-4, 1e-3, 1e-2, 1e-1],
    alphas=[0.7, 0.8, 0.9, 0.95, 0.99]
)

# Step 3: Architecture search with best params
arch_results = architecture_search(
    X_train, X_val,
    best_l1=best_params['l1_lambda'],
    best_alpha=best_params['alpha']
)

# Step 4: Final model training with best hyperparameters
final_model = train_final_model(
    X_train, X_val,
    l1_lambda=best_params['l1_lambda'],
    alpha=best_params['alpha'],
    embed_dim=64,
    n_layers=2,
    train_steps=5000
)

# Step 5: Final evaluation on test set (ONLY ONCE!)
test_metrics = evaluate_model(final_model, params, X_test)
print("Final Test Metrics:", test_metrics)

# Step 6: Save model and hyperparameters
save_model(final_model, params, best_params, "nnn_production_v1.pkl")
```

---

## 10. References

- NNN Paper Section 5: Model Selection via hyperparameter search
- Bergstra & Bengio (2012): Random Search for Hyper-Parameter Optimization
- Hyperparameter Importance Paper: Practical recommendations for gradient-based training
