# Stochastic Interest Rate Modelling: CIR, CIR++ and Jump-Diffusion

This project studies whether a single short-rate observable, the 3-month zero-coupon yield, can be used to predict a wider zero-coupon yield curve. The main notebook, `Stochastic_Interest_Rate_Modelling.ipynb`, builds and evaluates several stochastic interest-rate models from the Cox-Ingersoll-Ross family.

The work is written in Python and uses standard scientific-computing libraries such as NumPy, pandas, SciPy, statsmodels, scikit-learn and Matplotlib.

## Project Files

| File | Purpose |
|------|---------|
| `Stochastic_Interest_Rate_Modelling.ipynb` | Main notebook containing the full analysis, model calibration, prediction workflow, diagnostics and conclusions. |
| `train_data.csv` | Expected training dataset. Not included in this folder currently, but required by the notebook. |
| `test_data.csv` | Expected test dataset containing available test-period actual yields. Not included in this folder currently, but required by the notebook. |
| `test_data_3M.csv` | Expected test dataset containing only the 3M yield used as the prediction input. Not included in this folder currently, but required by the notebook. |

## Data Used

The notebook expects three CSV files:

| Dataset | Period | Rows | Role |
|---------|--------|------|------|
| `train_data.csv` | 2016-05-19 to 2024-04-26 | 1,976 | Used for cleaning, exploratory analysis and model calibration. |
| `test_data.csv` | 2024-04-29 to 2026-04-29 | 495 | Used only for post-hoc evaluation of predictions. |
| `test_data_3M.csv` | 2024-04-29 to 2026-04-29 | 495 | Used as the permitted test-time model input. |

The training data contains the following zero-coupon yield columns:

```text
Date, ZC025YR, ZC050YR, ZC075YR, ZC100YR, ZC200YR,
ZC500YR, ZC1000YR, ZC2000YR, ZC3000YR
```

The test data contains actual yields only up to 2 years:

```text
Date, ZC025YR, ZC050YR, ZC075YR, ZC100YR, ZC200YR
```

Because the test set does not contain 5Y, 10Y, 20Y or 30Y actual yields, the notebook generates predictions for those maturities but cannot quantitatively validate them out of sample.

## Problem Statement

The central constraint is that the prediction algorithm may ingest only the 3-month yield for a given test date. This 3M yield is treated as a proxy for the instantaneous short rate, \(r_t\).

Using that single input, the notebook predicts the yield curve for:

```text
6M, 9M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y
```

Predictions are evaluated where actual test yields are available: 6M, 9M, 1Y and 2Y.

## Implemented Formulas

### Data Cleaning and Time Steps

Calendar gaps are converted into variable year fractions:

```text
dt_i = (Date_i - Date_{i-1}) / 252
```

Outliers are detected using the Tukey IQR rule:

```text
IQR = Q3 - Q1
lower bound = Q1 - 2.0 * IQR
upper bound = Q3 + 2.0 * IQR
```

Flagged values are replaced by interpolation. Yields are clipped at `1e-5` to avoid invalid logarithms and square roots.

### CIR Short-Rate Process

The base CIR model is:

```text
dr_t = kappa(theta - r_t)dt + sigma * sqrt(r_t)dW_t
```

The Feller positivity condition is:

```text
2 * kappa * theta >= sigma^2
```

The mean-reversion half-life is:

```text
half-life = ln(2) / kappa
```

The conditional mean used for interpretation is:

```text
E[r_t] = r_0 * exp(-kappa * t) + theta * (1 - exp(-kappa * t))
```

The long-run variance is:

```text
Var[r_t] -> theta * sigma^2 / (2 * kappa)
```

### CIR Bond Pricing

For maturity `tau = T - t`, the zero-coupon bond price is:

```text
P(t,T) = A(tau) * exp(-B(tau) * r_t)
```

where:

```text
gamma = sqrt(kappa^2 + 2 * sigma^2)

B(tau) =
2 * (exp(gamma * tau) - 1)
/
((gamma + kappa) * (exp(gamma * tau) - 1) + 2 * gamma)

A(tau) =
[
  2 * gamma * exp((gamma + kappa) * tau / 2)
  /
  ((gamma + kappa) * (exp(gamma * tau) - 1) + 2 * gamma)
]^(2 * kappa * theta / sigma^2)
```

For numerical stability, the notebook computes `log_A` directly instead of raising `A` to a large exponent.

The model-implied yield is:

```text
y(t,tau) = (B(tau) * r_t - log(A(tau))) / tau
```

The prediction formula using the observed 3M rate is:

```text
y_hat(tau) =
(
  B(tau; kappa_hat, theta_hat, sigma_hat) * r_3M
  - log(A(tau; kappa_hat, theta_hat, sigma_hat))
)
/
tau
```

### CIR Simulation

The notebook simulates CIR paths with an Euler-Maruyama step:

```text
r_{t+dt} =
max(
  r_t + kappa(theta - r_t)dt + sigma * sqrt(r_t) * dW_t,
  1e-7
)
```

with:

```text
dW_t ~ Normal(0, dt)
```

### WLS/OLS Calibration

The discretised CIR equation is:

```text
dr ≈ kappa(theta - r_t)dt + sigma * sqrt(r_t) * epsilon
```

The regression form is:

```text
dr / dt = beta_0 + beta_1 * r_t + error
```

with parameter recovery:

```text
kappa = -beta_1
theta = beta_0 / kappa
```

The heteroskedastic CIR variance motivates weights proportional to:

```text
weight_t = 1 / r_t
```

The volatility estimate is:

```text
sigma =
std(
  residual / sqrt(r_t * dt)
)
```

If the raw estimate gives non-positive `kappa`, the notebook clamps it to `0.05` for downstream comparisons.

### NCX2 MLE Calibration

The exact CIR transition density is noncentral chi-square based. The notebook uses:

```text
exp_kdt = exp(-kappa * dt)
c = 2 * kappa / (sigma^2 * (1 - exp_kdt))
df = 4 * kappa * theta / sigma^2
u = c * exp_kdt * r_t
```

The negative log-likelihood objective is:

```text
NLL(kappa, theta, sigma) = -sum(log f(r_{t+dt} | r_t))
```

with a Feller penalty:

```text
penalty = 1e6 * max(0, sigma^2 - 2 * kappa * theta)
```

### Panel Least-Squares Calibration

Panel LS fits the model to the full training yield panel:

```text
loss(kappa, theta, sigma) =
mean(
  (Y_pred(kappa, theta, sigma) - Y_actual)^2
)
```

The implementation uses a volatility floor:

```text
sigma >= 0.05
```

and a hard Feller penalty:

```text
if 2 * kappa * theta < sigma^2:
    loss = 1e10 + 1e8 * (sigma^2 - 2 * kappa * theta)
```

### Dynamic CIR++

The shifted CIR++ short rate is:

```text
r_t++ = r_t + phi(t)
```

The yield correction is:

```text
y_hat++(tau) = y_hat_CIR(tau) + phi(tau)
```

The notebook implements a rate-adaptive deterministic shift:

```text
phi(tau, r_3M) = a_tau + b_tau * r_3M
```

For each maturity, the coefficients are fitted on training residuals:

```text
residual_tau = y_actual_tau - y_CIR_tau
residual_tau ≈ a_tau + b_tau * r_3M
```

Coefficients are then interpolated across maturities using cubic splines.

### Jump-Diffusion CIR

The JD-CIR model is:

```text
dr_t =
kappa(theta - r_t)dt
+ sigma * sqrt(r_t)dW_t
+ J_t dN_t
```

where:

```text
N_t = Poisson process with intensity lambda
J_t ~ Exponential(mu_J)
```

The jump model keeps the CIR `B(tau)` function unchanged and modifies `log_A`:

```text
log_A_JD(tau) =
log_A_CIR(tau)
+ lambda * integral_0^tau log(1 / (1 + mu_J * B(s))) ds
```

The integral is evaluated by trapezoidal quadrature.

The JD-CIR yield is:

```text
y_JD(t,tau) =
(B(tau) * r_t - log_A_JD(tau)) / tau
```

The jump correction used in diagnostics is:

```text
jump correction =
(log_A_JD(tau) - log_A_CIR(tau)) / tau
```

### Evaluation Metrics

Overall and per-maturity model quality is measured using:

```text
R^2 = 1 - sum((y_actual - y_pred)^2) / sum((y_actual - mean(y_actual))^2)
```

```text
RMSE = sqrt(mean((y_actual - y_pred)^2))
```

```text
MAE = mean(abs(y_actual - y_pred))
```

Errors are reported in basis points using:

```text
error_bps = error * 10000
```

The notebook also computes an empirical single-factor ceiling for a maturity using:

```text
R^2 ceiling = corr(r_3M, y_tau)^2
```

## Workflow

The notebook follows this pipeline:

1. Load and inspect the training and test data.
2. Clean yields by handling formatting issues, outliers, missing values and non-trading-day gaps.
3. Run stationarity diagnostics on the 3M short-rate series using ADF and KPSS tests.
4. Perform exploratory data analysis of yield tenors and term-structure behaviour.
5. Define the CIR short-rate model and its zero-coupon bond pricing equations.
6. Calibrate CIR parameters using three methods:
   - WLS/OLS from the 3M time series.
   - Noncentral chi-square transition-density MLE.
   - Panel least squares across the full training yield curve.
7. Compare calibration methods out of sample.
8. Predict the test-period yield curve from 3M only.
9. Extend the base CIR model using:
   - Dynamic CIR++ with rate-adaptive deterministic shifts.
   - Jump-diffusion CIR using the Duffie-Pan-Singleton affine jump-diffusion structure.
10. Run diagnostics, residual analysis, rolling R-squared checks and final model comparison.
11. Discuss regime shift, model limitations and references.

## Models Implemented

### Base CIR

The Cox-Ingersoll-Ross model is defined as:

```text
dr_t = kappa(theta - r_t)dt + sigma * sqrt(r_t)dW_t
```

where:

| Parameter | Meaning |
|-----------|---------|
| `kappa` | Mean-reversion speed. |
| `theta` | Long-run mean level. |
| `sigma` | Short-rate volatility. |

The notebook enforces the Feller condition:

```text
2 * kappa * theta >= sigma^2
```

This helps keep rates positive in the CIR process.

### Calibration Methods

The notebook compares three CIR calibration strategies:

| Method | Data Used | Result |
|--------|-----------|--------|
| WLS/OLS | 3M time series | Works reasonably but is weak for identifying the term-structure shape. |
| NCX2 MLE | 3M transition density | Captures time-series volatility but performs poorly for yield-curve prediction. |
| Panel LS | Cross-sectional yield panel | Best base-CIR method because it directly optimises yield-curve fit. |

Panel LS is used as the base prediction model because it gives the strongest out-of-sample yield-curve performance among the three base calibrations.

### Dynamic CIR++

The Dynamic CIR++ extension adds a rate-adaptive residual correction:

```text
correction_tau(r_t) = a_tau + b_tau * r_t
```

The coefficients are fitted on training residuals by maturity, then interpolated across maturities using cubic splines. This keeps the test-time input constraint intact because the correction still depends only on the 3M rate.

In this notebook, Dynamic CIR++ performs worse out of sample than the base CIR model, suggesting the residual correction overfits the training-period curve shape and does not transfer well to the test regime.

### JD-CIR

The jump-diffusion CIR extension follows the affine jump-diffusion idea from Duffie-Pan-Singleton. It adds rare jump shocks to the short-rate process while preserving tractable bond pricing.

The model is calibrated on a high-rate sub-sample beginning in 2022, which better matches the 2024-2026 test regime than the full 2016-2024 training window.

## Key Results

Final out-of-sample evaluation:

| Model | R-squared | RMSE |
|-------|----------:|-----:|
| JD-CIR (Duffie-Pan-Singleton) | 0.9426 | 16.05 bps |
| Base CIR (Panel LS) | 0.8959 | 21.61 bps |
| Dynamic CIR++ | 0.6224 | 41.15 bps |

The best overall model is:

```text
JD-CIR (Duffie-Pan-Singleton)
```

It meets the target out-of-sample R-squared threshold of 0.85.

Per-maturity results for the best model:

| Maturity Column | Maturity | R-squared |
|-----------------|----------|----------:|
| `ZC050YR` | 0.50Y | 0.9933 |
| `ZC075YR` | 0.75Y | 0.9772 |
| `ZC100YR` | 1.00Y | 0.9451 |
| `ZC200YR` | 2.00Y | 0.7107 |

The 2Y prediction is weaker because the test period contains an inverted curve regime. The notebook estimates the test-period single-factor R-squared ceiling for predicting 2Y from 3M at about 0.8245, so the lower 2Y performance is partly structural rather than just an implementation issue.

## Main Findings

- The base CIR model can predict short maturities well when calibrated with panel least squares.
- Calibration choice matters strongly. Panel LS performs much better for yield-curve prediction than time-series-only OLS or MLE.
- Dynamic CIR++ does not improve performance in this experiment and appears sensitive to regime shift.
- JD-CIR is the best model overall, improving both R-squared and RMSE.
- The test period differs materially from the training period: the training curve is mostly upward-sloping, while the test curve is inverted.
- Predictions for 5Y, 10Y, 20Y and 30Y are generated, but they cannot be validated because the test file does not contain those actual maturities.

## How to Run

1. Place the required CSV files in the same folder as `Stochastic_Interest_Rate_Modelling.ipynb`:

```text
train_data.csv
test_data.csv
test_data_3M.csv
```

2. Install the required Python packages:

```bash
pip install numpy pandas matplotlib scipy statsmodels scikit-learn
```

3. Open and run the notebook:

```bash
jupyter notebook Stochastic_Interest_Rate_Modelling.ipynb
```

or:

```bash
jupyter lab Stochastic_Interest_Rate_Modelling.ipynb
```

## Important Notes

- The repository currently contains only the notebook. The CSV input files are required to rerun the analysis.
- `test_data.csv` is used only for evaluation, not as a model input.
- `test_data_3M.csv` supplies the only permitted test-time predictor: the 3M yield.
- The notebook contains many plots for EDA, calibration sensitivity, predicted-vs-actual curves, residual diagnostics and rolling performance.
- Some long-maturity outputs are prediction-only because actual test yields are unavailable beyond 2Y.

## References

The notebook cites the following core references:

- Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985). "A theory of the term structure of interest rates." Econometrica.
- Duffie, D., Pan, J., Singleton, K. (2000). "Transform analysis and asset pricing for affine jump-diffusions." Econometrica.
- Brigo, D., Mercurio, F. (2006). Interest Rate Models: Theory and Practice.
- Longstaff, F.A., Schwartz, E.S. (1992). "Interest rate volatility and the term structure: a two-factor general equilibrium model." Journal of Finance.
