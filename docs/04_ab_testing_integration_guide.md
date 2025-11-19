# A/B Testing Integration Guide for NNN Marketing Mix Modeling

## Overview

This guide explains how to integrate A/B test results with the NNN framework to validate model predictions, calibrate attribution estimates, and improve model accuracy through experimental data.

A/B tests provide **ground truth** for marketing effectiveness and are essential for validating MMM outputs.

---

## 1. Why A/B Testing Matters for MMM

### 1.1 The Gold Standard for Validation

**A/B tests provide:**
- ✅ Causal evidence of marketing impact (not just correlation)
- ✅ Ground truth for incremental sales/conversions
- ✅ Channel-specific effectiveness benchmarks
- ✅ Calibration data for model predictions
- ✅ Detection of model bias and drift

**Without A/B tests:**
- ❌ MMM relies solely on observational data
- ❌ Attribution may reflect correlation, not causation
- ❌ No way to validate model accuracy
- ❌ Risk of overfitting to noise

---

### 1.2 Types of Marketing Experiments

| Experiment Type | Description | Use Case |
|----------------|-------------|----------|
| **Geo Holdout** | Turn off marketing in selected geos | Validate total channel impact |
| **Geo Lift** | Increase spend in selected geos | Measure incrementality |
| **Creative A/B Test** | Test different creatives with same budget | Validate creative quality embeddings |
| **Keyword Test** | Test keyword performance | Validate search embeddings |
| **Budget Test** | Test different spend levels | Validate saturation curves |
| **Timing Test** | Test different ad schedules | Validate temporal dynamics |

---

## 2. Integration Framework

### 2.1 Experimental Design Principles

**Key Requirements for Valid Tests:**

```python
# Checklist for valid A/B test design
ab_test_requirements = {
    'randomization': True,           # Random assignment to test/control
    'sufficient_sample_size': True,  # Statistical power >= 80%
    'sufficient_duration': True,     # At least 2-4 weeks
    'no_spillover': True,            # Control geos not affected by test
    'clean_implementation': True,    # No other changes during test
    'pre_period_balance': True       # Test/control similar before test
}
```

**Recommended Test Duration:**
- Minimum: 2 weeks (captures weekly patterns)
- Recommended: 4-6 weeks (accounts for delayed effects)
- Ideal: 8+ weeks (captures adstock and carryover)

---

### 2.2 Data Structure for A/B Tests

**A/B Test Data Schema:**

```python
ab_test_data = pd.DataFrame({
    'test_id': ['geo_holdout_001', 'geo_holdout_001', ...],
    'geo': ['US-CA', 'US-TX', ...],
    'group': ['control', 'test', ...],           # test = treatment
    'channel': ['YouTube', 'YouTube', ...],
    'start_date': ['2024-01-01', '2024-01-01', ...],
    'end_date': ['2024-02-01', '2024-02-01', ...],
    'spend_test': [0, 50000, ...],               # Spend during test
    'spend_control': [50000, 0, ...],            # Counterfactual spend
    'sales_test': [125000, 180000, ...],         # Observed sales
    'sales_control': [150000, 175000, ...],      # Control group sales
    'incremental_sales': [-25000, 5000, ...]     # Treatment effect
})
```

---

## 3. Validation Workflows

### 3.1 Geo Holdout Test Validation

**Scenario:** Turn off YouTube ads in 5 test geos for 4 weeks

**Step 1: Extract Test Period Data**

```python
def extract_test_period(X_tensor, geo_mapping, test_config):
    """
    Extract data for A/B test period.

    Args:
        X_tensor: Full data tensor (G, T, C, D)
        geo_mapping: Dict mapping geo names to indices
        test_config: Dict with test_id, start_week, end_week, channel_idx, test_geos

    Returns:
        X_test, X_control: Tensors for test and control groups
    """

    start_week = test_config['start_week']
    end_week = test_config['end_week']
    channel_idx = test_config['channel_idx']
    test_geos = test_config['test_geos']

    # Get geo indices
    test_geo_indices = [geo_mapping[geo] for geo in test_geos]
    control_geo_indices = [i for i in range(X_tensor.shape[0]) if i not in test_geo_indices]

    # Extract test period
    X_test_period = X_tensor[:, start_week:end_week, :, :]

    # Split by test/control
    X_test = X_test_period[test_geo_indices, :, :, :]
    X_control = X_test_period[control_geo_indices, :, :, :]

    return X_test, X_control, test_geo_indices, control_geo_indices


# Example: YouTube holdout test
test_config = {
    'test_id': 'youtube_holdout_q1_2024',
    'start_week': 52,   # Week 52 in data
    'end_week': 56,     # 4-week test
    'channel_idx': 2,   # YouTube channel
    'test_geos': ['US-CA', 'US-TX', 'US-FL', 'US-NY', 'US-IL']
}

X_test, X_control, test_idx, control_idx = extract_test_period(
    X_tensor, GEO_MAP, test_config
)
```

---

**Step 2: Calculate Observed Treatment Effect**

```python
def calculate_observed_effect(X_test, X_control, sales_idx=0):
    """
    Calculate observed incremental sales from A/B test.

    Returns:
        observed_lift: Total incremental sales (test vs. control)
    """

    # Extract sales (channel 0, dimension 0)
    sales_test = X_test[:, :, sales_idx, 0]      # (n_test_geos, n_weeks)
    sales_control = X_control[:, :, sales_idx, 0]  # (n_control_geos, n_weeks)

    # Average sales per geo
    avg_sales_test = np.mean(sales_test)
    avg_sales_control = np.mean(sales_control)

    # Total sales
    total_sales_test = np.sum(sales_test)
    total_sales_control = np.sum(sales_control)

    # Calculate lift
    observed_lift = total_sales_test - (total_sales_control * (len(sales_test) / len(sales_control)))

    print(f"Test geos avg sales: ${avg_sales_test:,.0f}")
    print(f"Control geos avg sales: ${avg_sales_control:,.0f}")
    print(f"Observed incremental sales: ${observed_lift:,.0f}")

    return {
        'observed_lift': observed_lift,
        'avg_sales_test': avg_sales_test,
        'avg_sales_control': avg_sales_control,
        'total_sales_test': total_sales_test,
        'total_sales_control': total_sales_control
    }

observed = calculate_observed_effect(X_test, X_control)
```

---

**Step 3: Calculate Model's Predicted Effect**

```python
def calculate_model_prediction(params, model, X_tensor, test_config, geo_mapping):
    """
    Calculate what the model predicts for the A/B test.

    Compares:
    - Factual: What actually happened (with treatment)
    - Counterfactual: What would have happened (without treatment)
    """

    # Get configuration
    start_week = test_config['start_week']
    end_week = test_config['end_week']
    channel_idx = test_config['channel_idx']
    test_geos = test_config['test_geos']
    test_geo_indices = [geo_mapping[geo] for geo in test_geos]

    # Factual prediction (with treatment)
    sales_factual, _ = model.apply(params, X_tensor)
    sales_factual_test = sales_factual[test_geo_indices, start_week:end_week]
    total_sales_factual = float(jnp.sum(sales_factual_test))

    # Counterfactual: Zero out the test channel in test geos during test period
    X_counterfactual = X_tensor.copy()
    for geo_idx in test_geo_indices:
        X_counterfactual = X_counterfactual.at[geo_idx, start_week:end_week, channel_idx, :].set(0)

    # Counterfactual prediction (without treatment)
    sales_counterfactual, _ = model.apply(params, X_counterfactual)
    sales_counterfactual_test = sales_counterfactual[test_geo_indices, start_week:end_week]
    total_sales_counterfactual = float(jnp.sum(sales_counterfactual_test))

    # Incremental sales = Factual - Counterfactual
    predicted_lift = total_sales_factual - total_sales_counterfactual

    print(f"Model predicted sales (with treatment): ${total_sales_factual:,.0f}")
    print(f"Model predicted sales (without treatment): ${total_sales_counterfactual:,.0f}")
    print(f"Model predicted lift: ${predicted_lift:,.0f}")

    return {
        'predicted_lift': predicted_lift,
        'total_sales_factual': total_sales_factual,
        'total_sales_counterfactual': total_sales_counterfactual
    }

predicted = calculate_model_prediction(params, nnn_model, X_tensor, test_config, GEO_MAP)
```

---

**Step 4: Compare and Validate**

```python
def validate_model_vs_test(observed, predicted, tolerance=0.20):
    """
    Compare model predictions against A/B test results.

    Args:
        observed: Dict from calculate_observed_effect
        predicted: Dict from calculate_model_prediction
        tolerance: Acceptable error margin (default 20%)

    Returns:
        validation_results: Dict with comparison metrics
    """

    obs_lift = observed['observed_lift']
    pred_lift = predicted['predicted_lift']

    # Calculate error
    absolute_error = pred_lift - obs_lift
    percentage_error = (absolute_error / obs_lift) * 100 if obs_lift != 0 else float('inf')

    # Validation check
    is_valid = abs(percentage_error) <= (tolerance * 100)

    results = {
        'observed_lift': obs_lift,
        'predicted_lift': pred_lift,
        'absolute_error': absolute_error,
        'percentage_error': percentage_error,
        'is_valid': is_valid,
        'tolerance': tolerance * 100
    }

    # Print results
    print("="*60)
    print("A/B TEST VALIDATION RESULTS")
    print("="*60)
    print(f"Observed incremental sales: ${obs_lift:,.0f}")
    print(f"Model predicted lift:       ${pred_lift:,.0f}")
    print(f"Absolute error:             ${absolute_error:,.0f}")
    print(f"Percentage error:           {percentage_error:.1f}%")
    print(f"")
    print(f"Tolerance threshold:        ±{tolerance*100:.0f}%")
    print(f"Validation status:          {'✅ PASS' if is_valid else '❌ FAIL'}")
    print("="*60)

    return results

validation = validate_model_vs_test(observed, predicted, tolerance=0.20)
```

**Visualization:**

```python
import plotly.graph_objects as go

def plot_validation_comparison(observed, predicted):
    """Visualize model vs. test comparison."""

    fig = go.Figure()

    # Observed vs Predicted
    fig.add_trace(go.Bar(
        x=['Observed (A/B Test)', 'Predicted (Model)'],
        y=[observed['observed_lift'], predicted['predicted_lift']],
        marker_color=['#2E86AB', '#A23B72'],
        text=[f"${observed['observed_lift']:,.0f}", f"${predicted['predicted_lift']:,.0f}"],
        textposition='outside'
    ))

    fig.update_layout(
        title='Model Validation: A/B Test vs. NNN Prediction',
        yaxis_title='Incremental Sales ($)',
        showlegend=False,
        height=400
    )

    fig.show()

plot_validation_comparison(observed, predicted)
```

---

### 3.2 Creative A/B Test Validation

**Scenario:** Test two different YouTube creatives with same budget

**Step 1: Extract Creative Performance from Test**

```python
def extract_creative_test_results(test_data):
    """
    Extract creative A/B test results.

    test_data: DataFrame with columns:
        - creative_id (e.g., 'Creative_A', 'Creative_B')
        - impressions
        - spend
        - conversions
        - sales
    """

    creative_A = test_data[test_data['creative_id'] == 'Creative_A']
    creative_B = test_data[test_data['creative_id'] == 'Creative_B']

    # Calculate metrics
    results = {
        'creative_A': {
            'total_sales': creative_A['sales'].sum(),
            'total_spend': creative_A['spend'].sum(),
            'roas': creative_A['sales'].sum() / creative_A['spend'].sum()
        },
        'creative_B': {
            'total_sales': creative_B['sales'].sum(),
            'total_spend': creative_B['spend'].sum(),
            'roas': creative_B['sales'].sum() / creative_B['spend'].sum()
        }
    }

    # Winner
    winner = 'creative_A' if results['creative_A']['roas'] > results['creative_B']['roas'] else 'creative_B'
    lift = abs(results['creative_A']['roas'] - results['creative_B']['roas']) / min(results['creative_A']['roas'], results['creative_B']['roas'])

    results['winner'] = winner
    results['roas_lift'] = lift

    return results
```

---

**Step 2: Validate Model's Creative Quality Ranking**

```python
def validate_creative_ranking(params, model, creative_A_embedding, creative_B_embedding,
                              X_base, channel_idx, test_week):
    """
    Validate that model correctly ranks creative quality.

    Args:
        creative_A_embedding: Embedding vector for Creative A
        creative_B_embedding: Embedding vector for Creative B
        X_base: Base input tensor
        channel_idx: Index of channel (e.g., YouTube)
        test_week: Week to test in
    """

    # Test Creative A
    X_test_A = X_base.copy()
    X_test_A = X_test_A.at[:, test_week, channel_idx, :].set(creative_A_embedding)
    sales_A, _ = model.apply(params, X_test_A)
    total_sales_A = float(jnp.sum(sales_A[:, test_week]))

    # Test Creative B
    X_test_B = X_base.copy()
    X_test_B = X_test_B.at[:, test_week, channel_idx, :].set(creative_B_embedding)
    sales_B, _ = model.apply(params, X_test_B)
    total_sales_B = float(jnp.sum(sales_B[:, test_week]))

    # Compare
    model_winner = 'A' if total_sales_A > total_sales_B else 'B'
    model_lift = abs(total_sales_A - total_sales_B) / min(total_sales_A, total_sales_B)

    return {
        'sales_A': total_sales_A,
        'sales_B': total_sales_B,
        'model_winner': model_winner,
        'model_lift': model_lift
    }
```

---

### 3.3 Multi-Test Calibration

**When you have multiple A/B tests:**

```python
def calibrate_across_tests(all_tests, params, model, X_tensor):
    """
    Calibrate model across multiple A/B tests.

    Args:
        all_tests: List of dicts, each containing:
            - test_config
            - observed_lift
    """

    results = []

    for test in all_tests:
        test_config = test['test_config']
        observed_lift = test['observed_lift']

        # Get model prediction
        predicted = calculate_model_prediction(params, model, X_tensor, test_config, GEO_MAP)
        pred_lift = predicted['predicted_lift']

        # Store
        results.append({
            'test_id': test_config['test_id'],
            'channel': test_config['channel_name'],
            'observed': observed_lift,
            'predicted': pred_lift,
            'error': pred_lift - observed_lift,
            'pct_error': ((pred_lift - observed_lift) / observed_lift) * 100
        })

    results_df = pd.DataFrame(results)

    # Calculate overall calibration
    mean_error = results_df['pct_error'].mean()
    rmse = np.sqrt(np.mean(results_df['error']**2))

    print(f"Overall calibration:")
    print(f"  Mean percentage error: {mean_error:.1f}%")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  Tests passing (<20% error): {(abs(results_df['pct_error']) < 20).sum()} / {len(results_df)}")

    return results_df

# Visualize calibration
import plotly.express as px

def plot_calibration(results_df):
    """Plot observed vs predicted across all tests."""

    fig = px.scatter(results_df,
                     x='observed',
                     y='predicted',
                     text='test_id',
                     color='channel',
                     title='Model Calibration: Observed vs. Predicted Lift',
                     labels={'observed': 'Observed Lift ($)', 'predicted': 'Predicted Lift ($)'})

    # Add diagonal line (perfect calibration)
    min_val = min(results_df['observed'].min(), results_df['predicted'].min())
    max_val = max(results_df['observed'].max(), results_df['predicted'].max())
    fig.add_scatter(x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', line=dict(dash='dash', color='gray'),
                    name='Perfect Calibration')

    fig.show()
```

---

## 4. Model Calibration and Correction

### 4.1 Bias Correction

**If model consistently over/under-predicts:**

```python
def calculate_calibration_factor(results_df):
    """
    Calculate bias correction factor from multiple tests.

    Args:
        results_df: DataFrame from calibrate_across_tests

    Returns:
        calibration_factor: Multiplicative correction factor
    """

    # Average ratio of observed to predicted
    calibration_factor = (results_df['observed'] / results_df['predicted']).mean()

    print(f"Calibration factor: {calibration_factor:.3f}")
    print(f"Interpretation: Model predictions should be multiplied by {calibration_factor:.3f}")

    return calibration_factor

# Apply calibration
def apply_calibration(predicted_lift, calibration_factor):
    """Apply bias correction to model predictions."""
    return predicted_lift * calibration_factor
```

---

### 4.2 Channel-Specific Calibration

**Different channels may require different calibrations:**

```python
def calculate_channel_calibration(results_df):
    """Calculate calibration factors per channel."""

    calibration = {}

    for channel in results_df['channel'].unique():
        channel_tests = results_df[results_df['channel'] == channel]
        factor = (channel_tests['observed'] / channel_tests['predicted']).mean()
        calibration[channel] = factor

        print(f"{channel}: {factor:.3f}")

    return calibration
```

---

## 5. Using A/B Tests to Improve Model Training

### 5.1 Incorporating Test Results as Training Signal

**Add A/B test results to loss function:**

```python
def loss_with_ab_test_penalty(params, X, model, ab_test_data):
    """
    Loss function incorporating A/B test validation.

    Args:
        ab_test_data: List of dicts with:
            - test_config
            - observed_lift
            - weight (importance of this test)
    """

    # Standard NNN loss
    sales_pred, search_pred = model.apply(params, X)
    sales_true = X[:, :, 0, 0]
    search_true = X[:, :, 1, :]

    loss_sales = jnp.mean((sales_true - sales_pred)**2)
    loss_search = jnp.mean((search_true - search_pred)**2)
    loss_standard = ALPHA * loss_sales + (1 - ALPHA) * loss_search

    # A/B test penalty
    ab_penalty = 0.0
    for test in ab_test_data:
        predicted = calculate_model_prediction(params, model, X, test['test_config'], GEO_MAP)
        pred_lift = predicted['predicted_lift']
        obs_lift = test['observed_lift']

        # Penalize deviation from test result
        test_error = (pred_lift - obs_lift)**2
        ab_penalty += test['weight'] * test_error

    # Combined loss
    total_loss = loss_standard + LAMBDA_AB * ab_penalty

    return total_loss, (loss_sales, loss_search, ab_penalty)

# Train with A/B test constraint
LAMBDA_AB = 0.1  # Weight for A/B test penalty
```

---

### 5.2 Holdout Validation Set from Tests

**Reserve test periods for validation:**

```python
def create_ab_validated_splits(X_tensor, ab_test_configs):
    """
    Create train/val/test splits that respect A/B test periods.

    Strategy:
    - Train: All data EXCEPT A/B test periods
    - Validation: A/B test periods
    - Test: Future holdout period
    """

    G, T, C, D = X_tensor.shape

    # Identify all test weeks
    test_weeks = set()
    for test in ab_test_configs:
        test_weeks.update(range(test['start_week'], test['end_week']))

    # Create masks
    train_weeks = [t for t in range(T) if t not in test_weeks]
    val_weeks = sorted(list(test_weeks))

    # Split
    X_train = X_tensor[:, train_weeks, :, :]
    X_val = X_tensor[:, val_weeks, :, :]

    print(f"Train weeks: {len(train_weeks)}")
    print(f"Validation weeks (from A/B tests): {len(val_weeks)}")

    return X_train, X_val
```

---

## 6. Best Practices

### 6.1 Experimental Design Recommendations

✅ **DO:**
- Run geo holdout tests every 6-12 months for major channels
- Test at least 2-4 weeks duration
- Use matched pair design (pair similar geos)
- Balance test/control groups on pre-period metrics
- Document all test details (dates, geos, channels, budgets)
- Share test results with modeling team immediately

❌ **DON'T:**
- Run tests during holiday periods (confounds results)
- Change test design mid-flight
- Use overlapping geos in multiple tests
- Run too many tests simultaneously (splits data too thin)
- Cherry-pick favorable test results

---

### 6.2 Validation Workflow Checklist

Before finalizing model:

- [ ] **At least 3 geo holdout tests** completed for major channels
- [ ] **Model predictions within ±20%** of observed test results
- [ ] **Creative A/B tests** validate embedding quality rankings
- [ ] **Calibration factors** calculated and applied if needed
- [ ] **Test results incorporated** into validation set
- [ ] **Systematic bias** checked and corrected
- [ ] **All test documentation** archived
- [ ] **Stakeholder alignment** on validation standards

---

## 7. Troubleshooting Common Issues

### Issue 1: Model Severely Overestimates Impact

**Symptoms:**
- Model predicts 2x-5x higher lift than A/B test
- Happens consistently across channels

**Likely Causes:**
- Overfitting to noise
- L1 regularization too weak
- Attribution window too long

**Solutions:**
```python
# Increase L1 regularization
l1_lambda = 0.05  # Increase from 0.01

# Reduce lookback window
lookback_window = 26  # Reduce from 52

# Add calibration
calibration_factor = 0.6  # Based on test results
```

---

### Issue 2: Model Underestimates Impact

**Symptoms:**
- Model predicts 50-80% of observed lift
- Missing delayed/carryover effects

**Likely Causes:**
- Lookback window too short
- Adstock decay too fast
- Missing indirect effects

**Solutions:**
```python
# Increase lookback window
lookback_window = 78  # Increase from 52

# Longer test duration to capture carryover
test_duration = 8  # weeks, not 4
```

---

### Issue 3: Creative Rankings Don't Match Tests

**Symptoms:**
- Model ranks Creative A > B, but test shows B > A

**Likely Causes:**
- Embedding quality doesn't capture effectiveness
- Volume effects dominating quality effects
- Insufficient training data for creative learning

**Solutions:**
```python
# Use A/B test results to fine-tune embeddings
# Manually adjust creative embeddings based on test results

# Or: Add creative test loss to training
# (See Section 5.1)
```

---

## 8. Example: Complete Validation Pipeline

```python
# Step 1: Load A/B test results
ab_tests = [
    {
        'test_id': 'youtube_holdout_q1',
        'test_config': {
            'start_week': 52,
            'end_week': 56,
            'channel_idx': 2,
            'channel_name': 'YouTube',
            'test_geos': ['US-CA', 'US-TX', 'US-FL']
        },
        'observed_lift': -45000,  # Negative because holdout (turned off)
        'weight': 1.0
    },
    {
        'test_id': 'search_lift_q2',
        'test_config': {
            'start_week': 78,
            'end_week': 82,
            'channel_idx': 3,
            'channel_name': 'SearchAds',
            'test_geos': ['US-NY', 'US-IL']
        },
        'observed_lift': 12000,  # Positive because lift test (increased spend)
        'weight': 1.0
    }
]

# Step 2: Validate model against all tests
results_df = calibrate_across_tests(ab_tests, params, nnn_model, X_tensor)

# Step 3: Calculate calibration factor
calibration_factor = calculate_calibration_factor(results_df)

# Step 4: Visualize
plot_calibration(results_df)

# Step 5: Apply corrections if needed
if abs(calibration_factor - 1.0) > 0.1:
    print(f"⚠️ Applying calibration factor: {calibration_factor:.3f}")
    # Retrain with A/B test penalty or apply post-hoc correction

# Step 6: Document validation
validation_report = f"""
A/B Test Validation Report
==========================

Model: NNN v1.0
Validation Date: 2024-01-15

Tests Evaluated: {len(ab_tests)}
Tests Passing (±20%): {(abs(results_df['pct_error']) < 20).sum()}
Overall RMSE: ${np.sqrt(np.mean(results_df['error']**2)):,.0f}
Calibration Factor: {calibration_factor:.3f}

Recommendation: {'✅ Model validated' if calibration_factor > 0.8 and calibration_factor < 1.2 else '❌ Recalibration required'}
"""

print(validation_report)
```

---

## 9. Integration with Production Systems

### 9.1 Continuous Validation

**Set up automated validation pipeline:**

```python
class ABTestValidator:
    """Continuous A/B test validation system."""

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.test_results = []

    def add_test_result(self, test_config, observed_lift):
        """Add new A/B test result for validation."""

        # Calculate model prediction
        predicted = calculate_model_prediction(
            self.params, self.model, X_tensor, test_config, GEO_MAP
        )

        # Store
        self.test_results.append({
            'test_id': test_config['test_id'],
            'date': pd.Timestamp.now(),
            'observed': observed_lift,
            'predicted': predicted['predicted_lift'],
            'error': predicted['predicted_lift'] - observed_lift,
            'pct_error': ((predicted['predicted_lift'] - observed_lift) / observed_lift) * 100
        })

        # Check if recalibration needed
        recent_tests = self.test_results[-5:]  # Last 5 tests
        avg_error = np.mean([t['pct_error'] for t in recent_tests])

        if abs(avg_error) > 25:
            print(f"⚠️ WARNING: Model drift detected. Average error: {avg_error:.1f}%")
            print("Recommendation: Retrain model with latest data")

        return predicted

    def generate_report(self):
        """Generate validation report."""
        df = pd.DataFrame(self.test_results)
        return df

# Usage
validator = ABTestValidator(nnn_model, params)

# Each time new A/B test completes
validator.add_test_result(new_test_config, observed_lift=-50000)

# Quarterly review
report = validator.generate_report()
print(report)
```

---

## 10. References and Further Reading

- **Google Causal Impact**: R package for causal inference with time series
- **Facebook Robyn**: Open-source MMM with experimental calibration
- **Kohavi, Tang, Xu (2020)**: "Trustworthy Online Controlled Experiments" - A/B testing best practices
- **NNN Paper Section 6**: Discusses validation approaches
- **Lewis & Rao (2015)**: "On the Near Impossibility of Measuring the Returns to Advertising" - Why A/B tests matter

---

## Summary

A/B testing is **essential** for validating MMM models. Key takeaways:

1. ✅ Run geo holdout tests every 6-12 months for major channels
2. ✅ Compare model predictions to test results (target: within ±20%)
3. ✅ Use test results to calibrate model or improve training
4. ✅ Validate creative rankings against creative A/B tests
5. ✅ Set up continuous validation pipeline
6. ✅ Document all tests and validation results

**Without A/B testing, you cannot trust your MMM model.**
