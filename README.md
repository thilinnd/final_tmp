# Statistical Arbitrage Trading System

## 🎯 Trading Models

### 1. Rolling OLS Model
**Process:**
- **Hedge Ratio**: OLS regression on rolling window (60 days)
- **Spread**: `Futures - (α + β × Stock)` where β is hedge ratio
- **Signal**: Z-score = `(Spread - Mean) / Std`
- **Long**: Z-score < -2.0 (Spread low → Long futures, Short stock)
- **Short**: Z-score > +2.0 (Spread high → Short futures, Long stock)
- **Exit**: |Z-score| < 0.5

### 2. Bayesian & Kalman Models
**Process:**
- **Model**: `Price_fut_t = α + β_t × Price_stock_t + ε_t` with β dynamic
- **Window**: Rolling window (60-120 days, suggest 60)
- **Prior**: β0 ~ Normal(1.0, 0.2²) (weakly informative)
- **Spread**: `Futures - (α + β × Stock)` where β is posterior mean
- **Uncertainty**: Bootstrap sampling (1000 samples) + Prior weighting
- **Signal Methods**: Z-score rule OR Posterior-predictive
- **Z-score Long**: Z-score < -2.0 AND Confidence > 0.3
- **Z-score Short**: Z-score > +2.0 AND Confidence > 0.3
- **Posterior-predictive**: P_revert ≥ 0.7 (spread ≤ 0 in 5 days)
- **Exit**: |Z-score| < 0.5 OR P_revert ≤ 0.5

## 📊 Trading Process


### Step 1: Model Estimation
```python
# Rolling OLS: Static hedge ratio per window (60 days)
hedge_ratio = OLS(futures_returns, stock_returns)  # futures = α + β*stock

# Bayesian/Kalman: Dynamic hedge ratio with uncertainty
# Model: Price_fut_t = α + β_t * Price_stock_t + ε_t
for window in rolling_windows(60):  # W = 60 days
    hedge_ratios = bootstrap_sampling(1000)
    # Apply prior: β0 ~ Normal(1.0, 0.2²)
    beta_prior = 1.0
    beta_std = 0.2
    hedge_ratio_mean = (mean(hedge_ratios) + 0.1 * beta_prior) / (1 + 0.1)
    uncertainty = std(hedge_ratios) / abs(hedge_ratio_mean)
```

### Step 2: Signal Generation
```python
# Calculate spread (corrected model)
spread = futures_price - (alpha + beta * stock_price)

# Method A: Z-score rule
z_score = (spread - rolling_mean) / rolling_std
if z_score < -2.0:  # Long spread
    action = "LONG_FUTURES_SHORT_STOCK"
elif z_score > +2.0:  # Short spread
    action = "SHORT_FUTURES_LONG_STOCK"
else:
    action = "EXIT"

# Method B: Posterior-predictive (recommended)
# P_revert = P(spread ≤ 0 in 5 days)
p_revert = simulate_future_spreads(beta_samples, horizon=5)
if p_revert >= 0.7:  # High probability of reversion
    action = "LONG_FUTURES_SHORT_STOCK"
elif p_revert <= 0.5:  # Low probability of reversion
    action = "EXIT"
```

### Step 3: Position Management
```python
# Rolling OLS: Fixed position size
position = base_size

# Bayesian/Kalman: Uncertainty-aware sizing
confidence = 1.0 - min(uncertainty, 0.8)
position = base_size * confidence * (1 - uncertainty)

# Position sizing based on signal method
if method == "z_score":
    position = base_size * (1 - abs(z_score) / 3.0)  # Reduce size for extreme z-scores
elif method == "posterior_predictive":
    position = base_size * p_revert  # Size based on reversion probability
```

## 🎯 Strategy Selection Formula

### Use Rolling OLS When:
```
Market_Volatility < 0.3 AND Data_Quality = "High" AND Risk_Tolerance = "High"
```

### Use Bayesian Kalman When:
```
Market_Volatility > 0.3 OR Uncertainty_Level > 0.5 OR Risk_Management = "Critical"
```

### Decision Matrix:
```
IF (Volatility < 0.3) AND (Uncertainty < 0.5) THEN Rolling_OLS
IF (Volatility > 0.3) OR (Uncertainty > 0.5) THEN Bayesian_Kalman
IF (Research_Mode = True) THEN Bayesian_Kalman
IF (Live_Trading = True) THEN Traditional_Kalman  # Best performance
```

### Quick Formulas:
- **Rolling OLS**: `Z-score < -2.0` → LONG, `Z-score > +2.0` → SHORT
- **Bayesian Kalman**: `Z-score < -1.5 AND Confidence > 0.3` → LONG 
