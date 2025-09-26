This repository contains the Python code developed for my Master Thesis in Financial Econometrics at Vrije Universiteit Amsterdam (May 2025).
The research evaluates which volatility model best forecasts stock market risk by applying a range of GARCH-family models and the Generalized Autoregressive Score (GAS) model to financial time series.

# Project Overview
The project investigates daily log-returns of two stocks from different sectors:
JPMorgan Chase & Co. (JPM) – banking sector
NVIDIA Corporation (NVDA) – technology/AI sector
Volatility is modeled using:
ARCH(1), GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1), and GAS(1,1) models
Three distributions: Normal, Student-t, and Skewed Student-t
Two estimation approaches:
Frequentist: Maximum Likelihood Estimation (MLE)
Bayesian: Random-Walk Metropolis–Hastings (RWMH)

# Key Features
- Parameter Estimation
  - MLE optimization for frequentist inference
  - Bayesian sampling using RWMH within an MCMC framework
- Model Evaluation
  - Value-at-Risk (VaR) backtesting (Kupiec UC, Christoffersen IND & CC tests)
  - Information criteria: Akaike (AIC) and Bayesian (BIC)
  - Loss functions: MSE, MAE, MAPE, RMSE
  - Logarithmic scoring rule
  - Diebold–Mariano (DM) predictive accuracy tests
- Distributional Flexibility
  - Normal, Student-t, and Skewed Student-t likelihoods to capture fat tails and skewness
- Visualization
  - Conditional variance and volatility plots
  - Return distributions and QQ plots
