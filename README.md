# Regression from first principles

This repo covers linear regression in two stages. First, I derive OLS by hand and solve it two ways: analytically and with gradient descent implemented in NumPy. Then I extend the same least-squares objective with ridge, lasso and elastic net regularisation on a multi-feature Bitcoin return model.

Each project has a notebook and a section for my handwritten derivations, so the maths sits next to the code it explains.

## Repository contents

| Project | Notebook | What it covers |
|---|---|---|
| OLS and gradient descent | `Gradient_Descent_OLS.ipynb` | Analytical OLS, NumPy gradient descent, standardisation and residual diagnostics |
| Regularised regression | `TimeSeries.ipynb` | Ridge, lasso and elastic net, time-ordered cross-validation and feature selection |

```text
.
├── README.md
├── Gradient_Descent_OLS.ipynb
├── TimeSeries.ipynb
└── handwritten/
    ├── ols/
    ├── gradient_descent/
    ├── standardisation/
    ├── ridge/
    ├── lasso/
    └── elastic_net/
```

## Setup

```bash
pip install jupyter numpy pandas matplotlib seaborn scipy statsmodels scikit-learn requests yfinance
```

Both notebooks pull live data, so they need internet access and the sample can change between runs. Everything reported below comes from my saved notebook outputs.

---

# Project 1: OLS and gradient descent

I calculate the OLS coefficients directly, fit the same model with gradient descent, and compare the results after converting back to the original units. The example regresses Ethereum log returns on Bitcoin log returns.

## Data and model

The notebook requests 365 days of daily USD prices from CoinGecko, aligns BTC and ETH by date, and calculates log returns:

\[
r_t=\log(P_t)-\log(P_{t-1}).
\]

The model is

\[
\hat y_i=\alpha+\beta x_i,
\]

where \(x_i\) is the BTC return and \(y_i\) is the ETH return on the same date. This is a same-period relationship, not a forecast. The saved run has 367 observations.

## Implementation

### Analytical OLS

\[
\hat\beta=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2},
\qquad
\hat\alpha=\bar y-\hat\beta\bar x.
\]

### Gradient descent

I standardise x and y, then minimise mean squared error:

\[
J(\alpha,\beta)=\frac1n\sum_i(y_i-\alpha-\beta x_i)^2.
\]

With \(e_i=y_i-\alpha-\beta x_i\), the gradients are

\[
\frac{\partial J}{\partial\alpha}=-\frac2n\sum_i e_i,
\qquad
\frac{\partial J}{\partial\beta}=-\frac2n\sum_i x_i e_i.
\]

Both parameters are updated together from the same current values.

| Setting | Value |
|---|---:|
| Initial intercept and slope | 0 |
| Learning rate | 0.01 |
| Loss-change tolerance | 1e-10 |
| Maximum iterations | 10,000,000 |

It stops when \(|J_{t+1}-J_t|<10^{-10}\). The fitted coefficients are then converted back to return units:

\[
\beta_{\mathrm{original}}=\beta_s\frac{s_y}{s_x},
\qquad
\alpha_{\mathrm{original}}=\bar y+s_y\alpha_s-\beta_{\mathrm{original}}\bar x.
\]

## Results

| Method | Intercept | Slope |
|---|---:|---:|
| Analytical OLS | 0.0014 | 1.4098 |
| Gradient descent | 0.001450 | 1.409678 |

Gradient descent met the stopping condition after 483 iterations. The two methods agree closely. The analytical coefficients are currently rounded to four decimal places inside the calculation, and gradient descent stops at a finite tolerance, so they are not numerically identical.

## Diagnostics and scope

The notebook includes ADF tests, return autocorrelation plots, residual plots and Ljung-Box tests. The saved Ljung-Box p-values are 0.578 at lag 10 and 0.290 at lag 20.

This is an in-sample comparison of two ways of solving the same problem, not a test of predictive performance. The hand-coded robust standard error block still needs its covariance formula corrected, so I do not use those intervals to support any conclusions.

## Handwritten derivations

### OLS: loss function to normal equations

Expanded squared-error loss, gradient, normal equations and the full-column-rank condition, with matrix dimensions and the link to the scalar formulas above.

<!-- INSERT OLS DERIVATION PHOTOS HERE -->

### Gradient descent: gradients and updates

Chain-rule derivation of both gradients, the simultaneous update, and the loss-change stopping rule used in the notebook.

<!-- INSERT GRADIENT DESCENT PHOTOS HERE -->

### Standardisation and original-unit coefficients

Why averaging and feature scaling do different jobs, and the algebra for converting the standardised intercept and slope back to the original units.

<!-- INSERT STANDARDISATION PHOTOS HERE -->

---

# Project 2: Regularised regression on Bitcoin returns

Here I look at how regularisation changes a regression with several related features. I build return-based features from crypto and market data, then use time-ordered cross-validation to choose the penalty strength and the balance between L1 and L2. The final model is fitted with scikit-learn's `ElasticNetCV`. In the saved run, cross-validation picks the lasso end of the candidate grid and keeps four of thirteen features.

## Data and features

BTC prices and volume come from CryptoCompare. ETH, SPY, GLD, the US Dollar Index, VIX and the US 10-year Treasury yield come from Yahoo Finance via yfinance. The merged sample runs from 2 November 2020 to 22 April 2026, giving 1,366 observations and 13 features after differencing and lagging.

The target is the BTC log return. Features are ETH, SPY, GLD and DXY returns, VIX and Treasury yield changes, BTC volume log change, BTC return lags 1 to 5, and a 7-observation rolling BTC volatility. Dates missing in any series are dropped before returns are calculated, so returns span consecutive retained dates rather than a fixed calendar interval.

## Model and validation

I use a chronological 80/20 split, with 1,092 training and 274 test observations. `StandardScaler` is fitted on the training set and applied to the test set. Tuning uses five expanding `TimeSeriesSplit` folds over these L1 ratios:

```python
[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
```

The objective is

\[
\frac1{2n}\lVert y-a\mathbf1-X\beta\rVert_2^2
+\lambda\rho\lVert\beta\rVert_1
+\frac{\lambda(1-\rho)}2\lVert\beta\rVert_2^2,
\]

where scikit-learn's `alpha` is \(\lambda\), `l1_ratio` is \(\rho\), and the intercept is not penalised.

## Results

| Selected setting | Value |
|---|---:|
| Penalty strength | 0.001206 |
| L1 ratio | 1.00 |
| Nonzero coefficients | 4 of 13 |

| Retained feature | Coefficient |
|---|---:|
| ETH return | 0.030046 |
| SPY return | 0.002049 |
| BTC volume log change | 0.000715 |
| VIX change | -0.000696 |

| Dataset | RMSE | MAE | R² |
|---|---:|---:|---:|
| Training | 0.023081 | 0.015750 | 0.6608 |
| Held-out period | 0.014460 | 0.010607 | 0.7205 |

Features are standardised and the target is not, so each coefficient is the change in fitted BTC log return per one training standard deviation of that feature, holding the others fixed. These are conditional associations, not causal effects. The notebook also plots the coefficients and the held-out actual versus fitted returns.

## Limitations

- The model uses same-period ETH, SPY, VIX and volume information, so this is contemporaneous modelling, not a demonstrated advance forecast.
- The rolling volatility feature includes the current target return. It has a zero final coefficient here, but it needs shifting, and every predictor's availability time needs checking, before this is treated as a forecasting design.
- The scaler is fitted before the internal cross-validation splits. It excludes the final test set, but a pipeline that scales within each training fold would be cleaner. I would rerun tuning and evaluation after that change.
- The seven-observation volatility window is not necessarily seven calendar days.
- A higher held-out R² than training R² does not by itself rule out overfitting, since the periods can differ in variance and market conditions.
- There are no separate OLS and ridge baselines, no fixed data snapshot and no trading-cost analysis yet.

## Handwritten derivations

### Ridge regression

L2-penalised objective, gradient, normal equations and closed-form solution, with the intercept left unpenalised and the loss scaling stated.

<!-- INSERT RIDGE DERIVATION PHOTOS HERE -->

### Lasso regression

L1-penalised objective, the derivative away from zero, and the subgradient condition at zero that allows exact zero coefficients.

<!-- INSERT LASSO DERIVATION PHOTOS HERE -->

### Elastic net and the implementation

Combined penalty, the mapping to `alpha` and `l1_ratio`, why scaling matters for penalised coefficients, and how this connects to the lasso solution and retained features above.

<!-- INSERT ELASTIC NET NOTES PHOTOS HERE -->
