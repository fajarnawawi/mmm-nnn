# Extended Validation Guide for NNN Marketing Mix Modeling

## Overview

This guide provides comprehensive validation techniques to ensure the NNN model produces reliable, actionable insights. Validation goes beyond standard metrics to test causal assumptions, robustness, and business validity.

---

## 1. Validation Framework

### 1.1 Validation Levels

```
Level 1: Statistical Validation
    ├── Prediction accuracy
    ├── Residual analysis
    └── Goodness of fit

Level 2: Causal Validation
    ├── Attribution consistency
    ├── Holdout experiments
    └── Synthetic control tests

Level 3: Robustness Validation
    ├── Cross-validation
    ├── Sensitivity analysis
    └── Stress testing

Level 4: Business Validation
    ├── Expert review
    ├── Historical consistency
    └── ROI plausibility
```

---

## 2. Level 1: Statistical Validation

### 2.1 Prediction Accuracy

**Hold-out Test Set Evaluation:**

```python
def comprehensive_prediction_eval(model, params, X_test):
    """
    Comprehensive evaluation on test set.
    """

    # Get predictions
    sales_pred, search_pred = model.apply(params, X_test)

    # Extract true values
    sales_true = X_test[:, :, 0, 0]  # Sales channel, first dim
    search_true = X_test[:, :, 1, :]  # Search channel, all dims

    # National level metrics
    sales_true_national = jnp.mean(sales_true, axis=0)
    sales_pred_national = jnp.mean(sales_pred, axis=0)

    national_metrics = {
        'mse': float(jnp.mean((sales_true_national - sales_pred_national)**2)),
        'rmse': float(jnp.sqrt(jnp.mean((sales_true_national - sales_pred_national)**2))),
        'mae': float(jnp.mean(jnp.abs(sales_true_national - sales_pred_national))),
        'mape': float(jnp.mean(jnp.abs((sales_true_national - sales_pred_national) / (sales_true_national + 1e-8))) * 100),
        'r2': float(1 - jnp.sum((sales_true_national - sales_pred_national)**2) / jnp.sum((sales_true_national - jnp.mean(sales_true_national))**2))
    }

    # Geo-level metrics
    geo_metrics = []
    for g in range(X_test.shape[0]):
        geo_r2 = 1 - jnp.sum((sales_true[g] - sales_pred[g])**2) / jnp.sum((sales_true[g] - jnp.mean(sales_true[g]))**2)
        geo_mape = jnp.mean(jnp.abs((sales_true[g] - sales_pred[g]) / (sales_true[g] + 1e-8))) * 100

        geo_metrics.append({
            'geo': g,
            'r2': float(geo_r2),
            'mape': float(geo_mape)
        })

    return national_metrics, pd.DataFrame(geo_metrics)

# Run evaluation
national_metrics, geo_metrics = comprehensive_prediction_eval(model, params, X_test)

print("National Metrics:")
for k, v in national_metrics.items():
    print(f"  {k.upper()}: {v:.4f}")

print("\nGeo-Level Performance:")
print(geo_metrics.describe())
```

**Acceptance Criteria:**
- ✅ National MAPE < 10%
- ✅ National R² > 0.80
- ✅ 90% of geos have R² > 0.70
- ✅ No systematic over/under prediction

---

### 2.2 Residual Analysis

**Check for patterns in residuals:**

```python
def residual_analysis(sales_true, sales_pred, dates):
    """
    Analyze residuals for patterns and heteroscedasticity.
    """

    residuals = sales_true - sales_pred

    # 1. Temporal patterns
    fig = px.scatter(x=dates, y=residuals,
                     title='Residuals Over Time',
                     labels={'x': 'Date', 'y': 'Residual'},
                     trendline='lowess')
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.show()

    # 2. Distribution
    fig = px.histogram(residuals, nbins=50,
                       title='Distribution of Residuals')
    fig.add_vline(x=0, line_dash='dash', line_color='red')
    fig.show()

    # 3. Q-Q plot for normality
    from scipy import stats
    fig = go.Figure()
    qq = stats.probplot(residuals, dist="norm")
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', name='Normal', line=dict(color='red')))
    fig.update_layout(title='Q-Q Plot', xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
    fig.show()

    # 4. Heteroscedasticity test
    fig = px.scatter(x=sales_pred, y=np.abs(residuals),
                     title='Absolute Residuals vs. Predictions (Heteroscedasticity Check)',
                     labels={'x': 'Predicted Sales', 'y': 'Absolute Residual'},
                     trendline='ols')
    fig.show()

    # Statistical tests
    print("\n=== Residual Diagnostics ===")
    print(f"Mean Residual: {np.mean(residuals):.4f} (should be ~0)")
    print(f"Std Residual: {np.std(residuals):.4f}")

    # Durbin-Watson test for autocorrelation
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(residuals)
    print(f"Durbin-Watson: {dw:.4f} (should be ~2 for no autocorrelation)")

    # Ljung-Box test for autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print(f"\nLjung-Box Test (lag 10):")
    print(lb_result)

# Run residual analysis
residual_analysis(sales_true_national, sales_pred_national, dates)
```

**Red Flags:**
- ❌ Residuals show clear temporal trend
- ❌ Residuals not normally distributed (heavy tails)
- ❌ Increasing variance with prediction level
- ❌ Significant autocorrelation (DW far from 2)

---

### 2.3 Out-of-Sample Forecasting

**Test model's ability to forecast future weeks:**

```python
def rolling_forecast_test(model, params, X_full, forecast_horizon=4):
    """
    Test forecasting accuracy by rolling forward in time.

    Args:
        forecast_horizon: Number of weeks to forecast ahead
    """

    G, T, C, D = X_full.shape

    # Start from 80% of data, forecast remaining 20%
    train_size = int(T * 0.8)

    forecasts = []
    actuals = []

    for t in range(train_size, T - forecast_horizon):
        # Use data up to time t
        X_history = X_full[:, :t, :, :]

        # Forecast next `forecast_horizon` weeks
        # (This requires extending the model or using autoregressive prediction)

        for h in range(1, forecast_horizon + 1):
            # Predict t+h
            # ... implementation depends on your forecasting approach

            forecast_t = sales_pred[:, t + h - 1]
            actual_t = X_full[:, t + h, 0, 0]

            forecasts.append(float(jnp.mean(forecast_t)))
            actuals.append(float(jnp.mean(actual_t)))

    # Calculate forecast accuracy
    forecasts = np.array(forecasts)
    actuals = np.array(actuals)

    mape = np.mean(np.abs((actuals - forecasts) / (actuals + 1e-8))) * 100

    print(f"Forecast MAPE ({forecast_horizon}-week ahead): {mape:.2f}%")

    # Visualize
    fig = px.scatter(x=actuals, y=forecasts,
                     title=f'{forecast_horizon}-Week Ahead Forecast Accuracy',
                     labels={'x': 'Actual Sales', 'y': 'Forecasted Sales'},
                     trendline='ols')
    fig.add_trace(go.Scatter(x=[actuals.min(), actuals.max()],
                             y=[actuals.min(), actuals.max()],
                             mode='lines', name='Perfect Forecast', line=dict(dash='dash', color='red')))
    fig.show()

    return mape
```

---

## 3. Level 2: Causal Validation

### 3.1 Attribution Consistency Checks

**Sanity checks for attribution results:**

```python
def validate_attribution(attribution_dict, total_sales, spend_dict):
    """
    Validate that attribution results are reasonable.
    """

    issues = []

    # 1. Attribution should sum to reasonable portion of total sales
    total_attribution = sum(attribution_dict.values())
    attribution_rate = total_attribution / total_sales

    print(f"Total Attribution: ${total_attribution:,.0f}")
    print(f"Total Sales: ${total_sales:,.0f}")
    print(f"Attribution Rate: {attribution_rate * 100:.1f}%")

    if attribution_rate < 0.3:
        issues.append("⚠️ Warning: Attribution explains < 30% of sales. Model may be missing key drivers.")
    elif attribution_rate > 1.2:
        issues.append("⚠️ Warning: Attribution > 120% of sales. Model may be double-counting effects.")

    # 2. Check ROI (attribution / spend) is reasonable
    print("\n=== ROI by Channel ===")
    for channel, attr in attribution_dict.items():
        if channel in spend_dict:
            roi = attr / spend_dict[channel]
            print(f"{channel}: ROI = {roi:.2f}x")

            if roi < 0:
                issues.append(f"❌ {channel} has negative ROI - likely model error")
            elif roi > 20:
                issues.append(f"⚠️ {channel} has extremely high ROI ({roi:.1f}x) - verify this is realistic")

    # 3. Check for directional consistency with correlations
    print("\n=== Directional Consistency ===")
    for channel, attr in attribution_dict.items():
        # In real implementation, check correlation of spend with sales
        # Positive correlation should lead to positive attribution
        pass

    # 4. Report issues
    if issues:
        print("\n=== Validation Issues ===")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ All attribution checks passed!")

    return len(issues) == 0
```

---

### 3.2 Holdout Experiment Validation

**If you have historical experiments (geo tests, time-based tests):**

```python
def validate_against_experiments(model, params, X_data, experiment_results):
    """
    Compare model attribution to results from controlled experiments.

    Args:
        experiment_results: Dict with structure:
            {
                'YouTube_2023Q1': {
                    'channel': 'YouTube',
                    'test_geos': ['US-CA', 'US-NY'],
                    'control_geos': ['US-TX', 'US-FL'],
                    'test_period': (100, 113),  # week indices
                    'measured_lift': 25000,  # From experiment
                },
                ...
            }
    """

    comparison = []

    for exp_name, exp in experiment_results.items():
        print(f"\nValidating against experiment: {exp_name}")

        # Get model's predicted attribution for this channel
        channel_idx = CHANNEL_MAP[exp['channel']]

        # Create counterfactual
        X_counter = X_data.copy()
        start_week, end_week = exp['test_period']
        X_counter[:, start_week:end_week, channel_idx, :] = 0.0

        # Predict sales with and without channel
        sales_baseline, _ = model.apply(params, X_data)
        sales_counter, _ = model.apply(params, X_counter)

        # Calculate lift for test geos during test period
        test_geo_indices = [GEO_MAP[g] for g in exp['test_geos']]

        model_lift = float(
            jnp.sum(sales_baseline[test_geo_indices, start_week:end_week]) -
            jnp.sum(sales_counter[test_geo_indices, start_week:end_week])
        )

        measured_lift = exp['measured_lift']
        error = abs(model_lift - measured_lift)
        pct_error = (error / measured_lift) * 100

        comparison.append({
            'experiment': exp_name,
            'channel': exp['channel'],
            'measured_lift': measured_lift,
            'model_lift': model_lift,
            'error': error,
            'pct_error': pct_error
        })

        print(f"  Measured Lift: ${measured_lift:,.0f}")
        print(f"  Model Lift: ${model_lift:,.0f}")
        print(f"  Error: {pct_error:.1f}%")

    results_df = pd.DataFrame(comparison)

    # Overall validation
    mean_error = results_df['pct_error'].mean()
    print(f"\n=== Experiment Validation Summary ===")
    print(f"Mean Absolute % Error: {mean_error:.1f}%")

    if mean_error < 20:
        print("✅ Model closely matches experiments (< 20% error)")
    elif mean_error < 40:
        print("⚠️ Model moderately matches experiments (20-40% error)")
    else:
        print("❌ Model poorly matches experiments (> 40% error) - investigate")

    return results_df
```

---

### 3.3 Synthetic Data Recovery Test

**Test if model can recover known ground truth:**

```python
def synthetic_data_recovery_test():
    """
    Generate synthetic data with known attribution, test if model recovers it.
    """

    print("Running Synthetic Data Recovery Test...")

    # Generate data with known causal structure
    # (Use the synthetic data generation from the notebook)
    X_synthetic, true_attribution = create_synthetic_dataset_with_truth(...)

    # Train model
    model = train_model(X_synthetic)

    # Get model's attribution
    pred_attribution = run_attribution(model, X_synthetic)

    # Compare
    for channel in true_attribution:
        true_val = true_attribution[channel]
        pred_val = pred_attribution[channel]
        error = abs(pred_val - true_val) / true_val * 100

        print(f"{channel}:")
        print(f"  True: ${true_val:,.0f}")
        print(f"  Predicted: ${pred_val:,.0f}")
        print(f"  Error: {error:.1f}%")

    # This should be < 10% error if model is working correctly
```

---

## 4. Level 3: Robustness Validation

### 4.1 K-Fold Time Series Cross-Validation

```python
def time_series_cv(X_tensor, n_splits=5):
    """
    Perform expanding window cross-validation.
    """

    G, T, C, D = X_tensor.shape

    results = []

    for i in range(n_splits):
        # Expanding window: train on increasing amounts of data
        train_end = int(T * (0.5 + i * 0.1))  # 50%, 60%, 70%, 80%, 90%
        test_end = min(train_end + int(T * 0.1), T)

        X_train = X_tensor[:, :train_end, :, :]
        X_test = X_tensor[:, train_end:test_end, :, :]

        print(f"\nFold {i+1}: Train weeks 0-{train_end}, Test weeks {train_end}-{test_end}")

        # Train model
        model, params = train_model(X_train)

        # Evaluate
        metrics = evaluate_model(model, params, X_test)

        results.append({
            'fold': i+1,
            'train_weeks': train_end,
            'test_weeks': test_end - train_end,
            **metrics
        })

    results_df = pd.DataFrame(results)

    print("\n=== Cross-Validation Summary ===")
    print(results_df)
    print(f"\nMean Test R²: {results_df['r2'].mean():.3f} ± {results_df['r2'].std():.3f}")
    print(f"Mean Test MAPE: {results_df['mape'].mean():.1f}% ± {results_df['mape'].std():.1f}%")

    return results_df
```

---

### 4.2 Sensitivity Analysis

**Test how attribution changes with perturbations:**

```python
def sensitivity_analysis(model, params, X_base, channel_idx, perturbation_range=[-20, -10, 0, 10, 20]):
    """
    Test how sensitive attribution is to spend changes.
    """

    baseline_attr = get_attribution(model, params, X_base, channel_idx)

    results = []

    for pct_change in perturbation_range:
        # Perturb channel spend
        X_perturbed = X_base.copy()

        # Scale channel volume by (1 + pct_change/100)
        scale = 1 + (pct_change / 100)
        X_perturbed = X_perturbed.at[:, :, channel_idx, :].multiply(scale)

        # Get new attribution
        new_attr = get_attribution(model, params, X_perturbed, channel_idx)

        results.append({
            'spend_change_pct': pct_change,
            'attribution': new_attr,
            'attribution_change_pct': ((new_attr - baseline_attr) / baseline_attr) * 100
        })

    results_df = pd.DataFrame(results)

    # Visualize
    fig = px.line(results_df, x='spend_change_pct', y='attribution_change_pct',
                  title='Sensitivity: How Attribution Changes with Spend',
                  labels={'spend_change_pct': 'Spend Change (%)',
                         'attribution_change_pct': 'Attribution Change (%)'})
    fig.add_trace(go.Scatter(x=[-20, 20], y=[-20, 20],
                             mode='lines', name='Linear Response', line=dict(dash='dash')))
    fig.show()

    print("\nSensitivity Analysis Results:")
    print(results_df)

    # Check for saturation/diminishing returns
    if results_df.iloc[-1]['attribution_change_pct'] < results_df.iloc[-1]['spend_change_pct'] * 0.7:
        print("✅ Model shows diminishing returns at high spend (realistic)")
    else:
        print("⚠️ Model may be too linear - check for proper saturation modeling")

    return results_df
```

---

### 4.3 Stress Testing

**Test model under extreme scenarios:**

```python
def stress_test(model, params, X_base):
    """
    Test model behavior under extreme scenarios.
    """

    tests = {
        'Zero all marketing': lambda X: X.at[:, :, 1:, :].set(0.0),  # Zero everything except sales
        'Double all marketing': lambda X: X.at[:, :, 1:, :].multiply(2.0),
        'Remove top channel': lambda X: X.at[:, :, get_top_channel_idx(), :].set(0.0),
        'Seasonal shock': lambda X: apply_seasonal_shock(X),
    }

    results = []

    baseline_sales = get_total_sales(model, params, X_base)

    for test_name, transform in tests.items():
        print(f"\nRunning stress test: {test_name}")

        X_test = transform(X_base.copy())

        test_sales = get_total_sales(model, params, X_test)
        change_pct = ((test_sales - baseline_sales) / baseline_sales) * 100

        results.append({
            'test': test_name,
            'baseline_sales': baseline_sales,
            'test_sales': test_sales,
            'change_pct': change_pct
        })

        print(f"  Sales change: {change_pct:+.1f}%")

    return pd.DataFrame(results)
```

---

## 5. Level 4: Business Validation

### 5.1 Expert Review Checklist

**Questions for domain experts:**

- [ ] Do the attribution percentages align with business intuition?
- [ ] Are high-performing channels consistent with past analyses?
- [ ] Do ROI estimates match historical campaign performance?
- [ ] Are seasonal patterns correctly captured?
- [ ] Do channel interactions make business sense?
- [ ] Are there any surprising results that need explanation?

### 5.2 Historical Consistency

**Compare to previous MMM results:**

```python
def compare_to_previous_mmm(current_attribution, previous_attribution):
    """
    Compare NNN results to previous MMM (e.g., traditional regression).
    """

    comparison = pd.DataFrame({
        'channel': list(current_attribution.keys()),
        'current_nnn': list(current_attribution.values()),
        'previous_mmm': [previous_attribution.get(ch, 0) for ch in current_attribution.keys()]
    })

    comparison['difference'] = comparison['current_nnn'] - comparison['previous_mmm']
    comparison['pct_change'] = (comparison['difference'] / comparison['previous_mmm']) * 100

    print("\n=== Comparison to Previous MMM ===")
    print(comparison)

    # Flag large changes
    large_changes = comparison[abs(comparison['pct_change']) > 50]
    if len(large_changes) > 0:
        print("\n⚠️ Large changes detected (>50%):")
        print(large_changes[['channel', 'pct_change']])
        print("Investigate these differences before deploying new model.")

    return comparison
```

---

## 6. Validation Report Template

```markdown
# NNN Model Validation Report

## 1. Executive Summary
- Model Version: v1.0
- Training Period: 2021-01-01 to 2023-12-31
- Test Period: 2024-01-01 to 2024-06-30
- Overall Assessment: [PASS / CONDITIONAL PASS / FAIL]

## 2. Statistical Validation
- Test R²: 0.XX
- Test MAPE: XX.X%
- Residuals: [Normal / Non-normal]
- Autocorrelation: [None / Detected]

## 3. Causal Validation
- Attribution Consistency: [PASS / FAIL]
- Experiment Validation MAE: XX.X%
- Synthetic Recovery Error: XX.X%

## 4. Robustness Validation
- CV Mean R²: 0.XX ± 0.XX
- Sensitivity Test: [Appropriate / Too sensitive / Too stable]
- Stress Test: [All scenarios reasonable / Issues detected]

## 5. Business Validation
- Expert Review: [Approved / Concerns raised]
- Historical Consistency: [Aligned / Deviations explained]
- ROI Plausibility: [All channels reasonable / Flags raised]

## 6. Recommendations
- [Go-live decision]
- [Follow-up actions needed]
- [Monitoring plan]

## 7. Known Limitations
- [List limitations]

## 8. Sign-off
- Data Scientist: __________ Date: __________
- Marketing Lead: __________ Date: __________
- Analytics Director: __________ Date: __________
```

---

## 7. Validation Checklist

Before deploying model to production:

- [ ] Test set MAPE < 10%
- [ ] Test set R² > 0.80
- [ ] Residuals show no patterns
- [ ] Attribution sums to 30-120% of total sales
- [ ] All channel ROIs are positive
- [ ] ROIs are within 0.5x - 20x range
- [ ] Experiment validation error < 30%
- [ ] Cross-validation stable (CV std < 0.05)
- [ ] Sensitivity analysis shows saturation
- [ ] Stress tests yield reasonable results
- [ ] Expert review completed and approved
- [ ] Historical consistency validated
- [ ] Documentation complete
- [ ] Stakeholder sign-off obtained

---

## 8. Next Steps After Validation

If validation passes:
1. ✅ Deploy to production
2. ✅ Set up monitoring dashboards
3. ✅ Schedule quarterly re-validation
4. ✅ Plan A/B tests to validate recommendations

If validation fails:
1. ❌ Investigate failure modes
2. ❌ Revisit data quality
3. ❌ Retune hyperparameters
4. ❌ Consider model architecture changes
5. ❌ Repeat validation

---

## References

- NNN Paper Section 6: Validation methodology
- Pearl (2009): Causality - Models, Reasoning, and Inference
- Athey & Imbens (2017): The State of Applied Econometrics - Causality and Policy Evaluation
