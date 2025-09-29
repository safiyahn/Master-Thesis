# Master Thesis & Advanced Econometrics Projects
This repository contains the Python code developed for my Master Thesis in Financial Econometrics at Vrije Universiteit Amsterdam (May 2025). 
I added additional material, which is a project of my Masters course Advanced Econometrics. This project shows econometric modelling and visualisation skills across multiple datasets using both Python and R. 

While the Master Thesis focuses primarily on advanced statistical modelling and formal testing (with results mainly presented in tables), the Advanced Econometrics project provides a stronger representation of my data visualisation skills, featuring extensive use of R and Python plotting tools.

## Master Thesis - Volatility Modelling in Financial Markets
The Master Thesis evaluates which volatility model best forecasts stock market risk by applying a range of GARCH-family models and the Generalized Autoregressive Score (GAS) model to financial time series.

### Project Overview
The project investigates daily log-returns of two stocks from different sectors:
- JPMorgan Chase & Co. (JPM) – banking sector
- NVIDIA Corporation (NVDA) – technology/AI sector


Volatility is modeled using:
- ARCH(1), GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1), and GAS(1,1) models

  
Three distributions:
- Normal, Student-t, and Skewed Student-t
  
Two estimation approaches:
- Frequentist: Maximum Likelihood Estimation (MLE)
- Bayesian: Random-Walk Metropolis–Hastings (RWMH)

### Key Features
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

### Repository Structure
- Master Thesis Safiyah Niamat.pdf – Full written thesis, including methodology, empirical results, and appendices.
- Master Thesis Code Stats-Plots.py – Python code to generate descriptive statistics, data cleaning, and all graphs (price series, log-returns, QQ-plots, conditional variance plots).
- Master Thesis Code Skewed Student-t Dist.py - Part of the thesis Python code, focusing on the implementation of GARCH-family and GAS models under the Skewed Student-t distribution only.
  
## 
 
## Advanced Econometrics Assignment
The assignment follows rotation through different departments to solve real-world econometric problems.
The project showcases the use of R and Python for data cleaning and management, econometric modelling, risk forecasting, and advanced visualisation.

### Project Overview
During the assignment, we acted as junior econometricians at Spectra Motors and tackled three applied business cases:
1. Quality Assurance – Equipment Failure Times
- Goal: Analyse failure-time data of key vehicle components to detect changes in supplier quality and time-varying reliability.
- Methods:
  - Maximum Likelihood Estimation (MLE) of exponential failure-time models with covariates (road quality, trip length).
  - Dynamic filtering to capture temporal shifts in component quality.
- Key Results:
  - Identification of components showing long-term quality trends versus temporary supply issues.
  - Log-likelihood comparisons confirming significant supplier effects.
    
2. Finance – Returns and Volatilities
- Goal: Model and forecast the conditional volatility of Spectra stock returns and compare with competitors (Tesla, General Motors).
- Methods:
  - Standard GARCH, Robust GARCH, and Robust GARCH with Leverage Effect (to capture asymmetric volatility responses).
  - Maximum Likelihood parameter estimation, AIC/BIC model selection, and hypothesis testing for robust filters and leverage effects.
  - Value-at-Risk (VaR) forecasting and backtesting using simulation and parametric methods.
- Key Results:
  - Evidence that robust filtering improves volatility modelling.
  - Leverage effects confirmed for Spectra and competitors, highlighting asymmetric risk dynamics.
  - Accurate VaR forecasts evaluated with out-of-sample hit-rate tests.
    
3. Marketing and Sales – Campaign Optimisation and Dynamic Pricing
- Goal: Evaluate marketing campaign effectiveness for Spectra’s new self-driving feature and estimate causal price effects for a leather interior upgrade.
- Methods:
  - Nonlinear dynamic parameter models to estimate advertising adstock effects (Google Ads, YouTube).
  - Impulse Response Functions (IRF) to measure the long-term impact of marketing shocks.
  - Instrumental Variable (IV) regressions and Hausman-Durbin-Wu tests to correct for endogeneity in pricing.
- Key Results:
  - Quantified diminishing returns to marketing spend across channels.
  - Causal estimates of price elasticity guiding optimal pricing policy.
  - Simulated profit impacts of alternative pricing strategies.
 
### Key Features
- Multi-language Implementation
  - R for Quality Assurance and Marketing analysis (data cleaning, dynamic parameter estimation, IV regression, advanced plotting).
  - Python for Financial Volatility modelling (robust GARCH estimation, VaR forecasting, backtesting).
- Data Management
  - Collection, cleaning, and validation of failure-time, financial returns, and marketing datasets.
  - Automated, reproducible code for maximum transparency and repeatability.
- Visualisation
  - High-quality plots: failure-time distributions, dynamic quality trends, news-impact curves, volatility filters, marketing adstock and sales responses.
 
### Repository Structure
- Advanced Econometrics Report.pdf – Full report with methodology, results, and interpretation. 
- Advanced Econometrics code Python.py – Stand-alone Python script for the financial volatility (GARCH) analysis.
- Advanced Econometrics code R.R - Stand-alone R script for the quality assurance and marketing and price analysis.
- GA_datafiles_Quality_Assurance.zip – Data for component failure-time modelling.
- GA_datafiles_Marketing_and_Sales.zip – Data for marketing and pricing analysis.
