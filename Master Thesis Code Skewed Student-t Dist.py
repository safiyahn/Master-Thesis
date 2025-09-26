#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: safiyah
"""

import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from scipy.stats import jarque_bera, probplot
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.optimize import minimize, fmin_slsqp
from scipy.stats import multivariate_normal
from math import pi, gamma, sqrt
import timeit
import numdifftools as nd
from numpy.linalg import inv, pinv
from scipy import sqrt,exp
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import chi2
from scipy.stats import norm, t
from arch.univariate import SkewStudent
from scipy.special import erf, gamma, gammaln

#Get the stock returns for the past 10 years
tickers = ["NVDA"]
start_date = "2015-01-01"
end_date = "2024-12-31"

returns = pd.DataFrame()
for ticker in tickers:
    returns[ticker] = yf.download(ticker, start_date, end_date)["Close"]

#Calculating the log returns
log_returns = (np.log(returns / returns.shift(1)).dropna())*100

#Initial values for ARCH model:
dOmega_ini_ARCH    = 1
dAlpha_ini_ARCH    = 0.05
dNu_ini_ARCH       = 6
dLambda_ini_ARCH   = 1.0

vTheta_ini_ARCH = [dOmega_ini_ARCH, dAlpha_ini_ARCH, dNu_ini_ARCH, dLambda_ini_ARCH]

#Restrictions for ARCH model:
vOmega_bnds_ARCH     = (0, 15)
vAlpha_bnds_ARCH     = (0, 1)
vNu_bnds_ARCH        = (2.0001, 30)
vLambda_bnds_ARCH    = (0.0001, 10)

vTheta_bnds_ARCH = [vOmega_bnds_ARCH, vAlpha_bnds_ARCH, vNu_bnds_ARCH, vLambda_bnds_ARCH]

#Initial values for GARCH model:
dOmega_ini_GARCH    = 0.1
dAlpha_ini_GARCH    = 0.05
dBeta_ini_GARCH     = 0.90
dNu_ini_GARCH       = 6
dLambda_ini_GARCH   = 1.0

vTheta_ini_GARCH = [dOmega_ini_GARCH, dAlpha_ini_GARCH, dBeta_ini_GARCH, dNu_ini_GARCH, dLambda_ini_GARCH]

#Restrictions for GARCH model:
vOmega_bnds_GARCH     = (0, 1)
vAlpha_bnds_GARCH     = (0, 1)
vBeta_bnds_GARCH      = (0, 1)
vNu_bnds_GARCH        = (2.0001, 30)
vLambda_bnds_GARCH    = (0.0001, 10)

vTheta_bnds_GARCH = [vOmega_bnds_GARCH, vAlpha_bnds_GARCH, vBeta_bnds_GARCH, vNu_bnds_GARCH, vLambda_bnds_GARCH]

#Initial values for GJR-GARCH model:
dOmega_ini_GJR_GARCH    = 0.1
dAlpha_ini_GJR_GARCH    = 0.05
dBeta_ini_GJR_GARCH     = 0.80
dGamma_ini_GJR_GARCH    = 0.05
dNu_ini_GJR_GARCH       = 6
dLambda_ini_GJR_GARCH   = 1.0

vTheta_ini_GJR_GARCH = [dOmega_ini_GJR_GARCH, dAlpha_ini_GJR_GARCH, dBeta_ini_GJR_GARCH, 
                        dGamma_ini_GJR_GARCH, dNu_ini_GJR_GARCH, dLambda_ini_GJR_GARCH]

#Restrictions for GJR-GARCH model:
vOmega_bnds_GJR_GARCH   = (0, 1)
vAlpha_bnds_GJR_GARCH   = (0, 1)
vBeta_bnds_GJR_GARCH    = (0, 1)
vGamma_bnds_GJR_GARCH   = (0, 1)
vNu_bnds_GJR_GARCH      = (2.0001, 30)
vLambda_bnds_GJR_GARCH  = (0.0001, 10)

vTheta_bnds_GJR_GARCH = [vOmega_bnds_GJR_GARCH, vAlpha_bnds_GJR_GARCH, vBeta_bnds_GJR_GARCH, 
                         vGamma_bnds_GJR_GARCH, vNu_bnds_GJR_GARCH, vLambda_bnds_GJR_GARCH]

#Initial values for EGARCH model:
dOmega_ini_EGARCH   = 0.1
dAlpha_ini_EGARCH   = 0.1
dBeta_ini_EGARCH    = 0.9
dGamma_ini_EGARCH   = 0.0
dNu_ini_EGARCH      = 6
dLambda_ini_EGARCH  = 1.0

vTheta_ini_EGARCH = [dOmega_ini_EGARCH, dAlpha_ini_EGARCH, dBeta_ini_EGARCH, 
                     dGamma_ini_EGARCH, dNu_ini_EGARCH, dLambda_ini_EGARCH]

#Restrictions for EGARCH model:
vOmega_bnds_EGARCH  = (-10, 10)
vAlpha_bnds_EGARCH  = (0, 1)
vBeta_bnds_EGARCH   = (0, 1)
vGamma_bnds_EGARCH  = (-1, 1)
vNu_bnds_EGARCH     = (2.0001, 30)
vLambda_bnds_EGARCH = (0.0001, 10)

vTheta_bnds_EGARCH = [vOmega_bnds_EGARCH, vAlpha_bnds_EGARCH, vBeta_bnds_EGARCH, 
                      vGamma_bnds_EGARCH, vNu_bnds_EGARCH, vLambda_bnds_EGARCH]

#Initial values for GAS model:
dOmega_ini_GAS  = -0.1
dAlpha_ini_GAS  = 0.1
dBeta_ini_GAS   = 0.8
dNu_ini_GAS     = 6
dLambda_ini_GAS = 1.0

vTheta_ini_GAS = [dOmega_ini_GAS, dAlpha_ini_GAS, dBeta_ini_GAS, dNu_ini_GAS, dLambda_ini_GAS]

#Restrictions for GAS model:
vOmega_bnds_GAS = (-10, 10)
vAlpha_bnds_GAS = (0, 1)
vBeta_bnds_GAS  = (0, 1)
vNu_bnds_GAS    = (2.0001, 30)
vLambda_bnds_GAS = (0.0001, 10)

vTheta_bnds_GAS = [vOmega_bnds_GAS, vAlpha_bnds_GAS, vBeta_bnds_GAS, vNu_bnds_GAS, vLambda_bnds_GAS]

# Skewed Student-t Distribution Functions
def skewed_t_pdf(x, mu, sigma, nu, xi):
    # Calculate constants m and s as defined in the PDF
    m = (gamma((nu-1)/2) * sqrt(nu-2)) / (sqrt(pi) * gamma(nu/2)) * (xi - 1/xi)
    s = sqrt((xi**2 + 1/xi**2 - 1) - m**2)
    
    # Calculate the adjusted value
    z = (x - mu)/sigma
    adj_z = s * z + m
    
    # Calculate the indicator function
    I = 1 if adj_z >= 0 else -1
    
    # Calculate the PDF
    const = (2 / (xi + 1/xi)) * (gamma((nu+1)/2) / (gamma(nu/2) * sqrt(pi*(nu-2)))) / sigma
    kernel = (1 + (adj_z**2 / (nu-2)) * xi**(-2*I))**(-(nu+1)/2)
    
    return const * s * kernel

def skewed_t_logpdf(x, mu, sigma, nu, xi):
    # Add parameter checks
    if nu <= 2:
        return -np.inf  # Return negative infinity for invalid parameters
    if xi <= 0:
        return -np.inf
        
    try:
        # Calculate constants m and s
        gamma_nu_minus_1 = gamma((nu-1)/2)
        gamma_nu = gamma(nu/2)
        sqrt_nu_minus_2 = sqrt(nu-2)
        
        m = (gamma_nu_minus_1 * sqrt_nu_minus_2) / (sqrt(pi) * gamma_nu) * (xi - 1/xi)
        s_squared = (xi**2 + 1/xi**2 - 1) - m**2
        
        if s_squared <= 0:
            return -np.inf
            
        s = sqrt(s_squared)
        
        # Calculate the adjusted value
        z = (x - mu)/sigma
        adj_z = s * z + m
        
        # Calculate the indicator function
        I = 1 if adj_z >= 0 else -1
        
        # Calculate the log PDF components
        log_const = (np.log(2) - np.log(xi + 1/xi) + 
                   gammaln((nu+1)/2) - gammaln(nu/2) - 
                   0.5*np.log(pi*(nu-2)) - np.log(sigma))
        
        log_kernel_term = 1 + (adj_z**2 / (nu-2)) * xi**(-2*I)
        if log_kernel_term <= 0:
            return -np.inf
            
        log_kernel = -((nu+1)/2) * np.log(log_kernel_term)
        
        return log_const + np.log(s) + log_kernel
        
    except:
        return -np.inf  # Return negative infinity for any numerical errors

def skewed_t_ppf(alpha, nu, xi):
    if alpha < 0.5:
        return -xi * t.ppf(1 - alpha, df=nu) * sqrt((nu-2)/nu)
    else:
        return t.ppf(alpha, df=nu) / xi * sqrt((nu-2)/nu)

#Skewed student-t Distribution -loglikelihood and MLE
def minusloglikelihood_ARCH(vTheta, stock_returns): 
    dOmega = vTheta[0]
    dAlpha = vTheta[1]
    dNu    = vTheta[2]
    dLambda = vTheta[3]
    
    iT = len(stock_returns)
    vH = np.ones(iT)
    
    vH[0] = np.var(stock_returns)
    
    for t in range(1, iT):
        vH[t] = dOmega + dAlpha * stock_returns[t-1]**2
    
    # Calculate log-likelihood using skewed t distribution
    vLogPdf = np.zeros(iT)
    for t in range(iT):
        vLogPdf[t] = skewed_t_logpdf(stock_returns[t], 0, np.sqrt(vH[t]), dNu, dLambda)
    
    dminusloglikelihood_skewed_t = -np.sum(vLogPdf)
    
    return dminusloglikelihood_skewed_t

def maxloglikelihood_ARCH(stock_returns,vTheta_ini, vTheta_bnds):
    ML = minimize(minusloglikelihood_ARCH, vTheta_ini, args=(stock_returns,), 
                      method='Nelder-Mead', bounds=vTheta_bnds, options = {'disp' : True})
    return ML

def minusloglikelihood_GARCH(vTheta, stock_returns): 
    dOmega = vTheta[0]
    dAlpha = vTheta[1]
    dBeta  = vTheta[2]
    dNu    = vTheta[3]
    dLambda = vTheta[4]  
    
    iT = len(stock_returns)
    vH = np.ones(iT)
    
    vH[0] = np.var(stock_returns)
    
    for t in range(1, iT):
        vH[t] = dOmega + dAlpha * stock_returns[t-1]**2 + dBeta * vH[t-1]
    
    # Calculate log-likelihood using skewed t distribution
    vLogPdf = np.zeros(iT)
    for t in range(iT):
        vLogPdf[t] = skewed_t_logpdf(stock_returns[t], 0, np.sqrt(vH[t]), dNu, dLambda)
    
    dminusloglikelihood_skewed_t = -np.sum(vLogPdf)
    
    return dminusloglikelihood_skewed_t

def maxloglikelihood_GARCH(stock_returns,vTheta_ini, vTheta_bnds):
    ML = minimize(minusloglikelihood_GARCH, vTheta_ini, args=(stock_returns,), 
                      method='Nelder-Mead', bounds=vTheta_bnds, options = {'disp' : True})
    return ML

def minusloglikelihood_GJR_GARCH(vTheta, stock_returns): 
    dOmega = vTheta[0]
    dAlpha = vTheta[1]
    dBeta  = vTheta[2]
    dGamma = vTheta[3]
    dNu    = vTheta[4]
    dLambda = vTheta[5]
    
    iT = len(stock_returns)
    vH = np.ones(iT)
    
    vH[0] = np.var(stock_returns)
    
    for t in range(1, iT):
        vH[t] = dOmega + (dAlpha + dGamma * (stock_returns[t-1] < 0)) * stock_returns[t-1]**2 + dBeta * vH[t-1]
    
    # Calculate log-likelihood using skewed t distribution
    vLogPdf = np.zeros(iT)
    for t in range(iT):
        vLogPdf[t] = skewed_t_logpdf(stock_returns[t], 0, np.sqrt(vH[t]), dNu, dLambda)
    
    dminusloglikelihood_skewed_t = -np.sum(vLogPdf)
    
    return dminusloglikelihood_skewed_t


def maxloglikelihood_GJR_GARCH(stock_returns, vTheta_ini, vTheta_bnds):
    ML = minimize(minusloglikelihood_GJR_GARCH, vTheta_ini, args=(stock_returns,), 
                             method='Nelder-Mead', bounds=vTheta_bnds, options={'disp': True})
    return ML

def minusloglikelihood_EGARCH(vTheta, stock_returns): 
    dOmega = vTheta[0]
    dAlpha = vTheta[1]
    dBeta  = vTheta[2]
    dGamma = vTheta[3]
    dNu    = vTheta[4]
    dLambda = vTheta[5]
    
    iT = len(stock_returns)
    vH = np.ones(iT)
    
    vH[0] = np.var(stock_returns)
    
    for t in range(1, iT):
        vH[t] = np.exp(dOmega + dAlpha * np.abs(stock_returns[t-1])/np.sqrt(vH[t-1]) + dBeta * np.log(vH[t-1]) + dGamma * stock_returns[t-1]/np.sqrt(vH[t-1]))
    
    # Calculate log-likelihood using skewed t distribution
    vLogPdf = np.zeros(iT)
    for t in range(iT):
        vLogPdf[t] = skewed_t_logpdf(stock_returns[t], 0, np.sqrt(vH[t]), dNu, dLambda)
    
    dminusloglikelihood_skewed_t = -np.sum(vLogPdf)
    
    return dminusloglikelihood_skewed_t

def maxloglikelihood_EGARCH(stock_returns,vTheta_ini, vTheta_bnds):
    ML = minimize(minusloglikelihood_EGARCH, vTheta_ini, args=(stock_returns,), 
                      method='Nelder-Mead', bounds=vTheta_bnds, options = {'disp' : True})
    return ML

def minusloglikelihood_GAS(vTheta, stock_returns):
    dOmega = vTheta[0]
    dAlpha = vTheta[1]
    dBeta  = vTheta[2]
    dNu    = vTheta[3]
    dLambda = vTheta[4]
    
    iT = len(stock_returns)
    vH = np.ones(iT)
    vH[0] = np.var(stock_returns)
    
    for t in range(1, iT):
        score_t = ((dNu + 1) * stock_returns[t-1]**2 / ((dNu - 2) * vH[t-1] + stock_returns[t-1]**2) - 1)
        log_ht = np.log(vH[t-1])
        log_ht = dOmega + dBeta * log_ht + dAlpha * score_t
        vH[t] = np.exp(log_ht)
    
    # Calculate log-likelihood using skewed t distribution
    vLogPdf = np.zeros(iT)
    for t in range(iT):
        vLogPdf[t] = skewed_t_logpdf(stock_returns[t], 0, np.sqrt(vH[t]), dNu, dLambda)
    
    return -np.sum(vLogPdf)

def maxloglikelihood_GAS(stock_returns, vTheta_ini, vTheta_bnds):
    ML = minimize(minusloglikelihood_GAS, vTheta_ini, args=(stock_returns,), 
                         method='Nelder-Mead', bounds=vTheta_bnds, options={'disp': True})
    return ML

#Skewed student-t Distribution variations of GARCH models
def ARCH_model(stock_returns, vTheta_ML):
    dOmega = vTheta_ML[0]
    dAlpha = vTheta_ML[1]
    p = 1 #Lag order
    T = len(stock_returns)
    vH = np.zeros(T)
    
    for t in range(p, T):
        vH[t] = dOmega + dAlpha * stock_returns[t-1]**2

    return vH[p:]   
    
def GARCH_model(stock_returns, vTheta_ML):
    dOmega = vTheta_ML[0]
    dAlpha = vTheta_ML[1]
    dBeta  = vTheta_ML[2]
    p = 1 #Lag order
    T = len(stock_returns)
    vH = np.zeros(T)
    
    for t in range(p, T):
        vH[t] = dOmega + dAlpha * stock_returns[t-1]**2 + dBeta * vH[t-1]

    return vH[p:]

def GJR_GARCH_model(stock_returns, vTheta_ML):
    dOmega = vTheta_ML[0]
    dAlpha = vTheta_ML[1]
    dBeta  = vTheta_ML[2]
    dGamma = vTheta_ML[3]
    p = 1
    T = len(stock_returns)
    vH = np.zeros(T)
    vH[0] = np.var(stock_returns)

    for t in range(p, T):
        vH[t] = dOmega + (dAlpha + dGamma * (stock_returns[t-1] < 0)) * stock_returns[t-1]**2 + dBeta * vH[t-1]

    return vH[p:]

def EGARCH_model(stock_returns, vTheta_ML):
    dOmega = vTheta_ML[0]
    dAlpha = vTheta_ML[1]
    dBeta  = vTheta_ML[2]
    dGamma = vTheta_ML[3]
    p = 1
    T = len(stock_returns)
    vH = np.zeros(T)
    vH[0] = np.var(stock_returns)

    for t in range(p, T):
        vH[t] = np.exp(dOmega + dAlpha * np.abs(stock_returns[t-1])/np.sqrt(vH[t-1]) + dBeta * np.log(vH[t-1]) + dGamma * stock_returns[t-1]/np.sqrt(vH[t-1]))          
                
    return vH[p:]

def GAS_model(stock_returns, vTheta_ML):
    dOmega = vTheta_ML[0]
    dAlpha = vTheta_ML[1]
    dBeta  = vTheta_ML[2]
    dNu    = vTheta_ML[3]
    
    T = len(stock_returns)
    vH = np.ones(T)
    vH[0] = np.var(stock_returns)
    
    for t in range(1, T):
        score_t = ((dNu + 1) * stock_returns[t-1]**2 / ((dNu - 2) * vH[t-1] + stock_returns[t-1]**2) - 1)
        log_ht = np.log(vH[t-1])
        log_ht = dOmega + dBeta * log_ht + dAlpha * score_t
        vH[t] = np.exp(log_ht)

    return vH[1:]

#Robust Hessian Calculation for GJR-GARCH
def safe_hessian_calculation(params, returns):
    try:
        # Try with central differences and slightly larger step
        hess = nd.Hessian(minusloglikelihood_GJR_GARCH, method='central', step=1e-4)(params, returns)

        # Add small ridge regularization to help invertibility
        hess += 1e-8 * np.eye(len(params))

        if not np.any(np.isnan(hess)):
            return hess
    except:
        pass

    print("Using element-wise fallback Hessian")
    n = len(params)
    hess = np.zeros((n, n))

    for i in range(n):
        def f(x):
            p = params.copy()
            p[i] = x
            return minusloglikelihood_GJR_GARCH(p, returns)
        hess[i, i] = nd.Hessian(f)(params[i])

    grad = nd.Gradient(minusloglikelihood_GJR_GARCH)(params, returns)
    for i in range(n):
        for j in range(i+1, n):
            hess[i, j] = hess[j, i] = grad[i] * grad[j]

    return hess

#Robust Hessian Calculation for GAS
def safe_hessian_calculation_gas(params, returns):
    try:
        hess = nd.Hessian(minusloglikelihood_GAS, method='central', step=1e-4)(params, returns)
        hess += 1e-8 * np.eye(len(params))  # Add small ridge
        if not np.any(np.isnan(hess)) and np.all(np.isfinite(hess)):
            return hess
    except Exception as e:
        print("Standard Hessian failed for GAS:", e)

    print("Using diagonal fallback Hessian for GAS")
    hess = np.diag(np.ones(len(params))) * 1e-2  # Small diagonal values
    return hess


#Calculate AIC and BIC
def calculate_aic(minuslikelihood, k):
    aic = 2 * k + 2 * minuslikelihood
    return aic

def calculate_bic(minuslikelihood, k, n):
    bic = np.log(n) * k + 2 * minuslikelihood
    return bic

#Calculate Random walk Metropolis-Hastings method:
def rwmhm(vTheta_ML, mCovariance_MLE, stock_returns, model_type):
    iNdraws = 1100            # total number of draws
    iBurnIn = 100             # number of burn-in draws
    iNumAcceptance = 0

    # matrix for storing the accepted (and repeated) draws:
    mTheta = np.zeros((iNdraws, len(vTheta_ML)))

    # feasible initial value: theta_0 = the ML estimator:
    mTheta[0] = vTheta_ML

    for i in range(1, iNdraws):
        # candidate draw: theta~ ~ N(theta_{i-1}, Sigma) with Sigma = cov(theta_ML):
        vTheta_candidate = multivariate_normal.rvs(mTheta[i-1], mCovariance_MLE)

        # if outside the allowed domain: reject immediately:
        if model_type == "ARCH":
            if (vTheta_candidate[0] < 0 or vTheta_candidate[1] < 0 or 
                vTheta_candidate[2] < 2 or vTheta_candidate[2] > 30 or
                vTheta_candidate[3] < 0.1):
                mTheta[i] = mTheta[i-1]
        elif model_type == "GARCH":
            if (vTheta_candidate[0] < 0 or vTheta_candidate[1] < 0 or 
                vTheta_candidate[2] < 0 or vTheta_candidate[3] < 2 or 
                vTheta_candidate[3] > 30 or vTheta_candidate[4] < 0.1):
                mTheta[i] = mTheta[i-1]
        elif model_type == "GJR-GARCH":
            if (vTheta_candidate[0] < 0 or vTheta_candidate[1] < 0 or 
                vTheta_candidate[2] < 0 or vTheta_candidate[3] < 0 or 
                vTheta_candidate[4] < 2 or vTheta_candidate[4] > 30 or
                vTheta_candidate[5] < 0.1):
                mTheta[i] = mTheta[i-1]
        elif model_type == "EGARCH":
            if (vTheta_candidate[4] < 2 or vTheta_candidate[4] > 30 or
                vTheta_candidate[5] < 0.1):
                mTheta[i] = mTheta[i-1]
        elif model_type == "GAS":
            if (vTheta_candidate[3] < 2 or vTheta_candidate[3] > 30 or
                vTheta_candidate[4] < 0.1):
                mTheta[i] = mTheta[i-1]
        else:
            raise ValueError("Invalid model type.")

        # if inside the allowed domain: compute acceptance probability and accept/reject candidate draw:
        # Note: minusloglikelihood functions give -loglikelihood:
        if model_type == "ARCH":
            dAccProb = min(
                np.exp(-minusloglikelihood_ARCH(vTheta_candidate, stock_returns) + minusloglikelihood_ARCH(mTheta[i-1], stock_returns)), 1
            )
        elif model_type == "GARCH":
            dAccProb = min(
                np.exp(-minusloglikelihood_GARCH(vTheta_candidate, stock_returns) + minusloglikelihood_GARCH(mTheta[i-1], stock_returns)), 1
            )
        elif model_type == "GJR-GARCH":
            dAccProb = min(
                np.exp(-minusloglikelihood_GJR_GARCH(vTheta_candidate, stock_returns) + minusloglikelihood_GJR_GARCH(mTheta[i-1], stock_returns)), 1
            )
        elif model_type == "EGARCH":
            dAccProb = min(
                np.exp(-minusloglikelihood_EGARCH(vTheta_candidate, stock_returns) + minusloglikelihood_EGARCH(mTheta[i-1], stock_returns)), 1
            )
        elif model_type == "GAS":
            dAccProb = min(
                np.exp(-minusloglikelihood_GAS(vTheta_candidate, stock_returns) + minusloglikelihood_GAS(mTheta[i-1], stock_returns)), 1)
        else:
            raise ValueError("Invalid model type.")
            
        dU = np.random.uniform(0, 1)
        if dU < dAccProb:
            mTheta[i] = vTheta_candidate
            iNumAcceptance = iNumAcceptance + 1
        else:
            mTheta[i] = mTheta[i-1]

    # Compute acceptance percentage:
    dAcceptancePercentage = 100 * iNumAcceptance / iNdraws
    print(f"acceptance percentage {model_type}: ", dAcceptancePercentage)
    
    # Estimate posterior mean & stdev as sample mean & sample stdev
    # of draws after the burn-in:
    vTheta_posterior_mean = np.mean(mTheta[iBurnIn:], axis=0)
    vTheta_posterior_stdev = np.std(mTheta[iBurnIn:], axis=0)
    
    print(f"posterior means {model_type}:", vTheta_posterior_mean)
    print(f"posterior stdev {model_type}:", vTheta_posterior_stdev)
    
    #Calculate the predicted conditional variance using the model
    if model_type == "ARCH":
        predicted_vH = ARCH_model(stock_returns, vTheta_posterior_mean)
    elif model_type == "GARCH":
        predicted_vH = GARCH_model(stock_returns, vTheta_posterior_mean)
    elif model_type == "GJR-GARCH":
        predicted_vH = GJR_GARCH_model(stock_returns, vTheta_posterior_mean)
    elif model_type == "EGARCH":
        predicted_vH = EGARCH_model(stock_returns, vTheta_posterior_mean)
    elif model_type == "GAS":
        predicted_vH = GAS_model(stock_returns, vTheta_posterior_mean)
    else:
        raise ValueError("Invalid model type.")
    
    #Calculate the predicted volatility (square root of the conditional variance)
    predicted_volatility = np.sqrt(predicted_vH)
    print(f'Conditional variance {model_type}', predicted_volatility)
    
    # # Histograms of draws from posterior distribution (draws after burn-in):
    # plt.hist(mTheta[iBurnIn:, 0], bins=20)
    # plt.title(f"Posterior Distribution of Parameter Omega ({model_type})")
    # plt.show()
    
    # plt.hist(mTheta[iBurnIn:, 1], bins=20)
    # plt.title(f"Posterior Distribution of Parameter Alpha ({model_type})")
    # plt.show()
    
    # if model_type == "GARCH" or model_type == "GJR-GARCH" or model_type == "EGARCH":
    #     plt.hist(mTheta[iBurnIn:, 2], bins=20)
    #     plt.title(f"Posterior Distribution of Parameter Beta ({model_type})")
    #     plt.show()
    
    # if model_type == "GJR-GARCH" or model_type == "EGARCH":
    #     plt.hist(mTheta[iBurnIn:, 3], bins=20)
    #     plt.title(f"Posterior Distribution of Parameter Gamma ({model_type})")
    #     plt.show()
        
    return predicted_volatility


def vH_plot(color, model, model_type):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{model_type} Conditional Variance', color=color)
    ax1.plot(log_returns.index[1:], model, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.title(f"{ticker} {model_type} Model Conditional Variance")
    plt.show()        
    
def log_vH_plot(color2, log_returns, model, model_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Returns', color=color)
    ax.plot(log_returns.index, log_returns[ticker], color=color)
    ax.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax.twinx()
    
    ax2.set_ylabel(f'Conditional Variance {model_type}', color=color2)
    ax2.plot(log_returns.index[1:], model, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.tight_layout()
    plt.title(f"{ticker} Log Returns and Conditional Variance {model_type}")
    plt.show()
    
#Calculate Mean Squared Error (MSE)
def calculate_mse(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    return mse

#Calculate Mean Absolute Error (MAE)
def calculate_mae(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    return mae

#Calculate Mean Absolute Percentage Error (MAPE)
def calculate_mape(actual, predicted):
    mape = np.mean(np.abs((actual - predicted) / (actual + predicted))) *100
    return mape

#Calculate Root Mean Squared Error (RMSE)
def calculate_rmse(actual, predicted):
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return rmse

# VaR and Backtesting Functions
def calculate_var(volatility, dNu, dLambda, alpha=0.05):
    q = skewed_t_ppf(alpha, dNu, dLambda)
    return -volatility * q

def backtest_var(returns, var_estimates, alpha=0.05):
    violations = returns < -var_estimates
    n_violations = np.sum(violations)
    expected_violations = alpha * len(returns)
    violation_ratio = n_violations / expected_violations
    
    return {
        'n_violations': n_violations,
        'expected_violations': expected_violations,
        'violation_ratio': violation_ratio,
        'violations': violations
    }

def uc_test(violations, alpha=0.05):
    n = len(violations)
    x = np.sum(violations)
    p_hat = x / n
    lr_uc = 2 * (np.log((p_hat**x) * ((1-p_hat)**(n-x)) - 
                np.log((alpha**x) * ((1-alpha)**(n-x)))))
    p_value_uc = 1 - chi2.cdf(lr_uc, 1)
    return lr_uc, p_value_uc

def ind_test(violations):
    n00, n01, n10, n11 = 0, 0, 0, 0
    
    for i in range(1, len(violations)):
        if violations[i-1] == 0 and violations[i] == 0:
            n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1:
            n01 += 1
        elif violations[i-1] == 1 and violations[i] == 0:
            n10 += 1
        else:
            n11 += 1
            
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    if pi0 == 0 or pi1 == 0 or pi == 0 or pi == 1:
        return 0, 1  # If undefined, return trivial case
    
    lr_ind = 2 * (np.log((1-pi0)**n00 * pi0**n01 * (1-pi1)**n10 * pi1**n11) - 
                np.log((1-pi)**(n00+n10) * pi**(n01+n11)))
    p_value_ind = 1 - chi2.cdf(lr_ind, 1)
    return lr_ind, p_value_ind

def cc_test(violations, alpha=0.05):
    lr_uc, p_uc = uc_test(violations, alpha)
    lr_ind, p_ind = ind_test(violations)
    lr_cc = lr_uc + lr_ind
    p_value_cc = 1 - chi2.cdf(lr_cc, 2)
    return lr_cc, p_value_cc

def perform_backtesting(returns, volatility, dNu, dLambda, model_name, alpha=0.05):
    # Calculate VaR using skewed Student-t distribution
    var_estimates = calculate_var(volatility, dNu, dLambda, alpha)
    
    # Backtest VaR
    backtest_results = backtest_var(returns, var_estimates, alpha)
    
    # Perform statistical tests
    uc_stat, uc_p = uc_test(backtest_results['violations'], alpha)
    ind_stat, ind_p = ind_test(backtest_results['violations'])
    cc_stat, cc_p = cc_test(backtest_results['violations'], alpha)
    
    return {
        'model': model_name,
        'var_estimates': var_estimates,
        'n_violations': backtest_results['n_violations'],
        'expected_violations': backtest_results['expected_violations'],
        'violation_ratio': backtest_results['violation_ratio'],
        'uc_stat': uc_stat,
        'uc_p': uc_p,
        'ind_stat': ind_stat,
        'ind_p': ind_p,
        'cc_stat': cc_stat,
        'cc_p': cc_p
    }

def print_backtest_results(results):
    print(f"\nBacktest Results for {results['model']}:")
    print(f"Number of violations: {results['n_violations']} (expected: {results['expected_violations']:.1f})")
    print(f"Violation ratio: {results['violation_ratio']:.2f}")
    print(f"UC test statistic: {results['uc_stat']:.4f} (p-value: {results['uc_p']:.4f})")
    print(f"IND test statistic: {results['ind_stat']:.4f} (p-value: {results['ind_p']:.4f})")
    print(f"CC test statistic: {results['cc_stat']:.4f} (p-value: {results['cc_p']:.4f})")
    
# Logarithmic scoring rules and DM tests
def logarithmic_scoring_skewed_t(actual_returns, predicted_volatility, dNu, dLambda):
    # Skewed-t log PDF
    log_scores = np.zeros(len(actual_returns))
    for t in range(len(actual_returns)):
        log_scores[t] = skewed_t_logpdf(actual_returns[t], 0, predicted_volatility[t], dNu, dLambda)
    return np.mean(log_scores)

def diebold_mariano_test(actual, pred1, pred2, loss_type='MSE', h=1):
    if loss_type == 'MSE':
        d = (actual - pred1)**2 - (actual - pred2)**2
    elif loss_type == 'RMSE':
        d = np.sqrt((actual - pred1)**2) - np.sqrt((actual - pred2)**2)
    elif loss_type == 'MAE':
        d = np.abs(actual - pred1) - np.abs(actual - pred2)
    elif loss_type == 'MAPE':
        # Adding small constant to denominator to avoid division by zero
        epsilon = 1e-10
        d = (np.abs(actual - pred1) / (np.abs(actual) + epsilon)) - \
            (np.abs(actual - pred2) / (np.abs(actual) + epsilon))
    else:
        raise ValueError("loss_type must be 'MSE', 'RMSE', 'MAE', or 'MAPE'")
    
    n = len(d)
    d_mean = np.mean(d)
    # Newey-West variance estimator for h-step ahead forecasts
    gamma = [np.sum((d[:n-lag] - d_mean) * (d[lag:] - d_mean)) / n for lag in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        var_d = gamma[0] / n  # Fallback to simple variance if NW gives negative value
    
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

def compare_models_with_dm(actual, models_dict, loss_type='MSE'):
    model_names = list(models_dict.keys())
    n_models = len(model_names)
    results = []
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            model1 = model_names[i]
            model2 = model_names[j]
            pred1 = models_dict[model1]
            pred2 = models_dict[model2]
            
            dm_stat, p_value = diebold_mariano_test(actual, pred1, pred2, loss_type)
            results.append({
                'Model 1': model1,
                'Model 2': model2,
                'DM Statistic': dm_stat,
                'p-value': p_value,
                'Loss Function': loss_type
            })
    
    return pd.DataFrame(results)

for ticker in tickers:
    stock_returns = log_returns[ticker]
    print("\n=== Skewed Student-t Distribution Results ===") 
    
    #Estimate the Maximum Loglikelihoods
    vTheta_ML_ARCH = maxloglikelihood_ARCH(stock_returns, vTheta_ini_ARCH, vTheta_bnds_ARCH).x    
    vTheta_ML_GARCH = maxloglikelihood_GARCH(stock_returns, vTheta_ini_GARCH, vTheta_bnds_GARCH).x    
    vTheta_ML_GJR_GARCH = maxloglikelihood_GJR_GARCH(stock_returns, vTheta_ini_GJR_GARCH, vTheta_bnds_GJR_GARCH).x    
    vTheta_ML_EGARCH = maxloglikelihood_EGARCH(stock_returns, vTheta_ini_EGARCH, vTheta_bnds_EGARCH).x    
    vTheta_ML_GAS = maxloglikelihood_GAS(stock_returns, vTheta_ini_GAS, vTheta_bnds_GAS).x

    mHessianofMinusLogL_ARCH = nd.Hessian(minusloglikelihood_ARCH)(vTheta_ML_ARCH, stock_returns)
    mCovariance_MLE_ARCH = inv(mHessianofMinusLogL_ARCH)
    vStandardErrors_ARCH = np.sqrt(np.diag(mCovariance_MLE_ARCH))

    mHessianofMinusLogL_GARCH = nd.Hessian(minusloglikelihood_GARCH)(vTheta_ML_GARCH, stock_returns)
    mCovariance_MLE_GARCH = inv(mHessianofMinusLogL_GARCH)
    vStandardErrors_GARCH = np.sqrt(np.diag(mCovariance_MLE_GARCH))
    
    try:
        mHessianofMinusLogL_GJR_GARCH = safe_hessian_calculation(vTheta_ML_GJR_GARCH, stock_returns)
        mCovariance_MLE_GJR_GARCH = inv(mHessianofMinusLogL_GJR_GARCH)
        vStandardErrors_GJR_GARCH = np.sqrt(np.abs(np.diag(mCovariance_MLE_GJR_GARCH)))
    except Exception as e:
        print("Failed to calculate robust GJR-GARCH standard errors:", e)
        vStandardErrors_GJR_GARCH = np.full(len(vTheta_ML_GJR_GARCH), np.nan)
    
    #mHessianofMinusLogL_GJR_GARCH = nd.Hessian(minusloglikelihood_GJR_GARCH)(vTheta_ML_GJR_GARCH, stock_returns)
    #mCovariance_MLE_GJR_GARCH = inv(mHessianofMinusLogL_GJR_GARCH)
    #vStandardErrors_GJR_GARCH = np.sqrt(np.diag(mCovariance_MLE_GJR_GARCH))
    
    mHessianofMinusLogL_EGARCH = nd.Hessian(minusloglikelihood_EGARCH)(vTheta_ML_EGARCH, stock_returns)
    mCovariance_MLE_EGARCH = inv(mHessianofMinusLogL_EGARCH)
    vStandardErrors_EGARCH = np.sqrt(np.diag(mCovariance_MLE_EGARCH))
    
    
    #mHessianofMinusLogL_GAS = nd.Hessian(minusloglikelihood_GAS)(vTheta_ML_GAS, stock_returns)
    mHessianofMinusLogL_GAS = safe_hessian_calculation_gas(vTheta_ML_GAS, stock_returns)
    mCovariance_MLE_GAS = inv(mHessianofMinusLogL_GAS)
    vStandardErrors_GAS = np.sqrt(np.diag(mCovariance_MLE_GAS))
    
    np.set_printoptions(precision=4)
    
    print(f"\n=== ML estimator and standard errors {ticker} ===");
    print("ARCH model", np.stack( (vTheta_ML_ARCH, vStandardErrors_ARCH), axis=1 ));
    print("GARCH model", np.stack( (vTheta_ML_GARCH, vStandardErrors_GARCH), axis=1 ));
    print("GJR-GARCH model", np.stack( (vTheta_ML_GJR_GARCH, vStandardErrors_GJR_GARCH), axis=1 ));
    print("EGARCH model", np.stack( (vTheta_ML_EGARCH, vStandardErrors_EGARCH), axis=1 ));
    print("GAS model", np.stack( (vTheta_ML_GAS, vStandardErrors_GAS), axis=1 ));

    #Compute the conditional variance of the GARCH models
    print("\n=== Conditional Variance Results ===") 
    ARCH_skewed_t = ARCH_model(stock_returns,vTheta_ML_ARCH)
    print("ARCH model conditional variance:", ARCH_skewed_t)
    
    GARCH_skewed_t = GARCH_model(stock_returns,vTheta_ML_GARCH)
    print("GARCH model conditional variance:", GARCH_skewed_t)
    
    GJR_GARCH_skewed_t = GJR_GARCH_model(stock_returns,vTheta_ML_GJR_GARCH)
    print("GJR-GARCH model conditional variance:", GJR_GARCH_skewed_t)
    
    EGARCH_skewed_t = EGARCH_model(stock_returns,vTheta_ML_EGARCH)
    print("EGARCH model conditional variance:", EGARCH_skewed_t)
    
    GAS_skewed_t = GAS_model(stock_returns, vTheta_ML_GAS)
    print("GAS model conditional variance:", GAS_skewed_t)
    
    #Calculate MSE, MAE, MAPE, RMSE: MLE
    print("\n=== Loss Functions Results MLE ===") 
    mse_ARCH = calculate_mse(stock_returns[1:]**2, ARCH_skewed_t)
    print("Mean Squared Error ARCH model with MLE:", mse_ARCH)
    mae_ARCH = calculate_mae(stock_returns[1:]**2, ARCH_skewed_t)
    print("Mean Absolute Error ARCH model with MLE:", mae_ARCH)
    mape_ARCH = calculate_mape(stock_returns[1:]**2, ARCH_skewed_t)
    print("Mean Absolute Percentage Error ARCH model with MLE:", mape_ARCH)
    rmse_ARCH = calculate_rmse(stock_returns[1:]**2, ARCH_skewed_t)
    print("Root Mean Squared Error ARCH model with MLE:", rmse_ARCH)
    
    mse_GARCH = calculate_mse(stock_returns[1:]**2, GARCH_skewed_t)
    print("Mean Squared Error GARCH model with MLE:", mse_GARCH)
    mae_GARCH = calculate_mae(stock_returns[1:]**2, GARCH_skewed_t)
    print("Mean Absolute Error GARCH model with MLE:", mae_GARCH)
    mape_GARCH = calculate_mape(stock_returns[1:]**2, GARCH_skewed_t)
    print("Mean Absolute Percentage Error GARCH model with MLE:", mape_GARCH)
    rmse_GARCH = calculate_rmse(stock_returns[1:]**2, GARCH_skewed_t)
    print("Root Mean Squared Error GARCH model with MLE:", rmse_GARCH)
    
    mse_GJR_GARCH = calculate_mse(stock_returns[1:]**2, GJR_GARCH_skewed_t)
    print("Mean Squared Error GJR-GARCH model with MLE:", mse_GJR_GARCH)
    mae_GJR_GARCH = calculate_mae(stock_returns[1:]**2, GJR_GARCH_skewed_t)
    print("Mean Absolute Error GJR-GARCH model with MLE:", mae_GJR_GARCH)
    mape_GJR_GARCH = calculate_mape(stock_returns[1:]**2, GJR_GARCH_skewed_t)
    print("Mean Absolute Percentage Error GJR-GARCH model with MLE:", mape_GJR_GARCH)
    rmse_GJR_GARCH = calculate_rmse(stock_returns[1:]**2, GJR_GARCH_skewed_t)
    print("Root Mean Squared Error GARCH model with MLE:", rmse_GJR_GARCH)
    
    mse_EGARCH = calculate_mse(stock_returns[1:]**2, EGARCH_skewed_t)
    print("Mean Squared Error EGARCH model with MLE:", mse_EGARCH)
    mae_EGARCH = calculate_mae(stock_returns[1:]**2, EGARCH_skewed_t)
    print("Mean Absolute Error EGARCH model with MLE:", mae_EGARCH)
    mape_EGARCH = calculate_mape(stock_returns[1:]**2, EGARCH_skewed_t)
    print("Mean Absolute Percentage Error EGARCH model with MLE:", mape_EGARCH)
    rmse_EGARCH = calculate_rmse(stock_returns[1:]**2, EGARCH_skewed_t)
    print("Root Mean Squared Error GARCH model with MLE:", rmse_EGARCH)
    
    mse_GAS = calculate_mse(stock_returns[1:]**2, GAS_skewed_t)
    print("Mean Squared Error GAS model with MLE:", mse_GAS)
    mae_GAS = calculate_mae(stock_returns[1:]**2, GAS_skewed_t)
    print("Mean Absolute Error GAS model with MLE:", mae_GAS)
    mape_GAS = calculate_mape(stock_returns[1:]**2, GAS_skewed_t)
    print("Mean Absolute Percentage Error GAS model with MLE:", mape_GAS)
    rmse_GAS = calculate_rmse(stock_returns[1:]**2, GAS_skewed_t)
    print("Root Mean Squared Error GARCH model with MLE:", rmse_GAS)
    
    # #Plot the conditional variances separately: MLE
    # vH_plot('tab:red', ARCH_skewed_t, "ARCH")
    # vH_plot('tab:green', GARCH_skewed_t, "GARCH")
    # vH_plot('tab:orange', GJR_GARCH_skewed_t, "GJR-GARCH")
    # vH_plot('tab:purple', EGARCH_skewed_t, "EGARCH")
    # vH_plot('tab:brown', GAS_skewed_t, "GAS")
    
    #Plot the log returns and conditional variance: MLE
    log_vH_plot('tab:red' , log_returns, ARCH_skewed_t, "ARCH")
    log_vH_plot('tab:green', log_returns, GARCH_skewed_t, "GARCH")
    log_vH_plot('tab:orange', log_returns, GJR_GARCH_skewed_t, "GJR-GARCH")
    log_vH_plot('tab:purple', log_returns, EGARCH_skewed_t, "EGARCH")
    log_vH_plot('tab:pink', log_returns, GAS_skewed_t, "GAS")
    
    #Random walk Metropolis-Hastings method:
    print(f"\n=== Random walk Metropolis-Hastings method {ticker} ===")
    rwmhm_ARCH = rwmhm(vTheta_ML_ARCH, mCovariance_MLE_ARCH, stock_returns, "ARCH")
    rwmhm_GARCH = rwmhm(vTheta_ML_GARCH, mCovariance_MLE_GARCH, stock_returns, "GARCH")
    rwmhm_GJR_GARCH = rwmhm(vTheta_ML_GJR_GARCH, mCovariance_MLE_GJR_GARCH, stock_returns, "GJR-GARCH")
    rwmhm_EGARCH = rwmhm(vTheta_ML_EGARCH, mCovariance_MLE_EGARCH, stock_returns, "EGARCH")
    rwmhm_GAS = rwmhm(vTheta_ML_GAS, mCovariance_MLE_GAS, stock_returns, "GAS")
    
    #Calculate MSE, MAE and MAPE: RWMHM
    print("\n=== Loss Functions Results RWMHM ===")    
    mse_ARCH_rwmhm = calculate_mse(stock_returns[1:]**2, rwmhm_ARCH)
    print("Mean Squared Error ARCH model with RWMHM:", mse_ARCH_rwmhm)
    mae_ARCH_rwmhm = calculate_mae(stock_returns[1:]**2, rwmhm_ARCH)
    print("Mean Absolute Error ARCH model with RWMHM:", mae_ARCH_rwmhm)
    mape_ARCH_rwmhm = calculate_mape(stock_returns[1:]**2, rwmhm_ARCH)
    print("Mean Absolute Percentage Error ARCH model with RWMHM:", mape_ARCH_rwmhm)
    rmse_ARCH_rwmhm = calculate_rmse(stock_returns[1:]**2, rwmhm_ARCH)
    print("Root Mean Squared Error ARCH model with MLE:", rmse_ARCH_rwmhm)
    
    mse_GARCH_rwmhm = calculate_mse(stock_returns[1:]**2, rwmhm_GARCH)
    print("Mean Squared Error GARCH model with RWMHM:", mse_GARCH_rwmhm)
    mae_GARCH_rwmhm = calculate_mae(stock_returns[1:]**2, rwmhm_GARCH)
    print("Mean Absolute Error GARCH model with RWMHM:", mae_GARCH_rwmhm)
    mape_GARCH_rwmhm = calculate_mape(stock_returns[1:]**2, rwmhm_GARCH)
    print("Mean Absolute Percentage Error GARCH model with RWMHM:", mape_GARCH_rwmhm)
    rmse_GARCH_rwmhm = calculate_rmse(stock_returns[1:]**2, rwmhm_GARCH)
    print("Root Mean Squared Error GARCH model with MLE:", rmse_GARCH_rwmhm)
    
    mse_GJR_GARCH_rwmhm = calculate_mse(stock_returns[1:]**2, rwmhm_GJR_GARCH)
    print("Mean Squared Error GJR-GARCH model with RWMHM:", mse_GJR_GARCH_rwmhm)
    mae_GJR_GARCH_rwmhm = calculate_mae(stock_returns[1:]**2, rwmhm_GJR_GARCH)
    print("Mean Absolute Error GJR-GARCH model with RWMHM:", mae_GJR_GARCH_rwmhm)
    mape_GJR_GARCH_rwmhm = calculate_mape(stock_returns[1:]**2, rwmhm_GJR_GARCH)
    print("Mean Absolute Percentage Error GJR-GARCH model with RWMHM:", mape_GJR_GARCH_rwmhm)
    rmse_GJR_GARCH_rwmhm = calculate_rmse(stock_returns[1:]**2, rwmhm_GJR_GARCH)
    print("Root Mean Squared Error GARCH model with MLE:", rmse_GJR_GARCH_rwmhm)
    
    mse_EGARCH_rwmhm = calculate_mse(stock_returns[1:]**2, rwmhm_EGARCH)
    print("Mean Squared Error EGARCH model with RWMHM:", mse_EGARCH_rwmhm)
    mae_EGARCH_rwmhm = calculate_mae(stock_returns[1:]**2, rwmhm_EGARCH)
    print("Mean Absolute Error EGARCH model with RWMHM:", mae_EGARCH_rwmhm)
    mape_EGARCH_rwmhm = calculate_mape(stock_returns[1:]**2, rwmhm_EGARCH)
    print("Mean Absolute Percentage Error EGARCH model with RWMHM:", mape_EGARCH_rwmhm)
    rmse_EGARCH_rwmhm = calculate_rmse(stock_returns[1:]**2, rwmhm_EGARCH)
    print("Root Mean Squared Error GARCH model with MLE:", rmse_EGARCH_rwmhm)
    
    mse_GAS_rwmhm = calculate_mse(stock_returns[1:]**2, rwmhm_GAS)
    print("Mean Squared Error GAS model with RWMHM:", mse_GAS_rwmhm)
    mae_GAS_rwmhm = calculate_mae(stock_returns[1:]**2, rwmhm_GAS)
    print("Mean Absolute Error GAS model with RWMHM:", mae_GAS_rwmhm)
    mape_GAS_rwmhm = calculate_mape(stock_returns[1:]**2, rwmhm_GAS)
    print("Mean Absolute Percentage Error GAS model with RWMHM:", mape_GAS_rwmhm)
    rmse_GAS_rwmhm = calculate_rmse(stock_returns[1:]**2, rwmhm_GAS)
    print("Root Mean Squared Error GARCH model with MLE:", rmse_GAS_rwmhm)
    
    #Plot the log returns and conditional variance: RWMHM
    log_vH_plot('tab:red' , log_returns, rwmhm_ARCH, "ARCH")
    log_vH_plot('tab:green', log_returns, rwmhm_GARCH, "GARCH")
    log_vH_plot('tab:orange', log_returns, rwmhm_GJR_GARCH, "GJR-GARCH")
    log_vH_plot('tab:purple', log_returns, rwmhm_EGARCH, "EGARCH")
    log_vH_plot('tab:pink', log_returns, rwmhm_GAS, "GAS")
    
    #Calculate AIC and BIC
    minuslikelihood_ARCH = -minusloglikelihood_ARCH(vTheta_ML_ARCH, stock_returns)
    minuslikelihood_GARCH = -minusloglikelihood_GARCH(vTheta_ML_GARCH, stock_returns)
    minuslikelihood_GJR_GARCH = -minusloglikelihood_GJR_GARCH(vTheta_ML_GJR_GARCH, stock_returns)
    minuslikelihood_EGARCH = -minusloglikelihood_EGARCH(vTheta_ML_EGARCH, stock_returns)
    loglikelihood_GAS = -minusloglikelihood_GAS(vTheta_ML_GAS, stock_returns)

    print("\n=== Loglikelihood Results ===")    
    print(f"ARCH model skewed student-t loglikelihood {ticker}: ", minuslikelihood_ARCH);
    print(f"GARCH model skewed student-t loglikelihood {ticker}: ", minuslikelihood_GARCH);
    print(f"GJR-GARCH model skewed student-t loglikelihood {ticker}: ", minuslikelihood_GJR_GARCH);
    print(f"EGARCH model skewed student-t loglikelihood {ticker}: ", minuslikelihood_EGARCH);
    print(f"GAS model skewed student-t loglikelihood {ticker}: ", loglikelihood_GAS);

    k_ARCH = len(vTheta_ML_ARCH)
    k_GARCH = len(vTheta_ML_GARCH)
    k_GJR_GARCH = len(vTheta_ML_GJR_GARCH)
    k_EGARCH = len(vTheta_ML_EGARCH)
    k_GAS = len(vTheta_ML_GAS)
    n = len(stock_returns)
    
    print("\n=== AIC & BIC Results ===")
    aic_ARCH = calculate_aic(-minuslikelihood_ARCH, k_ARCH)
    bic_ARCH = calculate_bic(-minuslikelihood_ARCH, k_ARCH, n)
    print(f"ARCH model AIC for {ticker}: ", aic_ARCH)
    print(f"ARCH model BIC for {ticker}: ", bic_ARCH)
    
    aic_GARCH = calculate_aic(-minuslikelihood_GARCH, k_GARCH)
    bic_GARCH = calculate_bic(-minuslikelihood_GARCH, k_GARCH, n)
    print(f"GARCH model AIC for {ticker}: ", aic_GARCH)
    print(f"GARCH model BIC for {ticker}: ", bic_GARCH)
    
    aic_GJR_GARCH = calculate_aic(-minuslikelihood_GJR_GARCH, k_GJR_GARCH)
    bic_GJR_GARCH = calculate_bic(-minuslikelihood_GJR_GARCH, k_GJR_GARCH, n)
    print(f"GJR-GARCH model AIC for {ticker}: ", aic_GJR_GARCH)
    print(f"GJR-GARCH model  BIC for {ticker}: ", bic_GJR_GARCH)
    
    aic_EGARCH = calculate_aic(-minuslikelihood_EGARCH, k_EGARCH)
    bic_EGARCH = calculate_bic(-minuslikelihood_EGARCH, k_EGARCH, n)
    print(f"EGARCH model AIC for {ticker}: ", aic_EGARCH)
    print(f"EGARCH model  BIC for {ticker}: ", bic_EGARCH)
    
    aic_GAS = calculate_aic(-loglikelihood_GAS, k_GAS)
    bic_GAS = calculate_bic(-loglikelihood_GAS, k_GAS, n)
    print(f"GAS model AIC for {ticker}: ", aic_GAS)
    print(f"GAS model BIC for {ticker}: ", bic_GAS)
    
    # Extract degrees of freedom
    dNu_ARCH = vTheta_ML_ARCH[2]
    dNu_GARCH = vTheta_ML_GARCH[3]
    dNu_GJR_GARCH = vTheta_ML_GJR_GARCH[4]
    dNu_EGARCH = vTheta_ML_EGARCH[4]
    dNu_GAS = vTheta_ML_GAS[3]
    
    # Extract degrees of lambda
    dLambda_ARCH = vTheta_ML_ARCH[3]
    dLambda_GARCH = vTheta_ML_GARCH[4]
    dLambda_GJR_GARCH = vTheta_ML_GJR_GARCH[5]
    dLambda_EGARCH = vTheta_ML_EGARCH[5]
    dLambda_GAS = vTheta_ML_GAS[4]
    
    #Calculate logarithmic scores for all models (MLE)
    print("\n=== Logarithmic Scoring Rules (MLE) ===")
    log_score_ARCH = logarithmic_scoring_skewed_t(stock_returns[1:], np.sqrt(ARCH_skewed_t), dNu_ARCH, dLambda_ARCH)
    log_score_GARCH = logarithmic_scoring_skewed_t(stock_returns[1:], np.sqrt(GARCH_skewed_t), dNu_GARCH, dLambda_GARCH)
    log_score_GJR_GARCH = logarithmic_scoring_skewed_t(stock_returns[1:], np.sqrt(GJR_GARCH_skewed_t), dNu_GJR_GARCH, dLambda_GJR_GARCH)
    log_score_EGARCH = logarithmic_scoring_skewed_t(stock_returns[1:], np.sqrt(EGARCH_skewed_t), dNu_EGARCH, dLambda_EGARCH)
    log_score_GAS = logarithmic_scoring_skewed_t(stock_returns[1:], np.sqrt(GAS_skewed_t),dNu_GAS, dLambda_GAS)
    
    print(f"ARCH (MLE) logarithmic score: {log_score_ARCH:.4f}")
    print(f"GARCH (MLE) logarithmic score: {log_score_GARCH:.4f}")
    print(f"GJR-GARCH (MLE) logarithmic score: {log_score_GJR_GARCH:.4f}")
    print(f"EGARCH (MLE) logarithmic score: {log_score_EGARCH:.4f}")
    print(f"GAS (MLE) logarithmic score: {log_score_GAS:.4f}")
    
    #Calculate logarithmic scores for all models (RWMHM)
    print("\n=== Logarithmic Scoring Rules (RWMHM) ===")
    log_score_ARCH_rwmhm = logarithmic_scoring_skewed_t(stock_returns[1:], rwmhm_ARCH, dNu_ARCH, dLambda_ARCH)
    log_score_GARCH_rwmhm = logarithmic_scoring_skewed_t(stock_returns[1:], rwmhm_GARCH, dNu_GARCH, dLambda_GARCH)
    log_score_GJR_GARCH_rwmhm = logarithmic_scoring_skewed_t(stock_returns[1:], rwmhm_GJR_GARCH, dNu_GJR_GARCH, dLambda_GJR_GARCH)
    log_score_EGARCH_rwmhm = logarithmic_scoring_skewed_t(stock_returns[1:], rwmhm_EGARCH, dNu_EGARCH, dLambda_EGARCH)
    log_score_GAS_rwmhm = logarithmic_scoring_skewed_t(stock_returns[1:], rwmhm_GAS, dNu_GAS, dLambda_GAS)

    print(f"ARCH (RWMHM) logarithmic score: {log_score_ARCH_rwmhm:.4f}")
    print(f"GARCH (RWMHM) logarithmic score: {log_score_GARCH_rwmhm:.4f}")
    print(f"GJR-GARCH (RWMHM) logarithmic score: {log_score_GJR_GARCH_rwmhm:.4f}")
    print(f"EGARCH (RWMHM) logarithmic score: {log_score_EGARCH_rwmhm:.4f}")
    print(f"GAS (RWMHM) logarithmic score: {log_score_GAS_rwmhm:.4f}")
    
    # Perform backtesting for all models (MLE)
    backtest_results_ARCH = perform_backtesting(stock_returns[1:], np.sqrt(ARCH_skewed_t), dNu_ARCH, dLambda_ARCH, "ARCH (MLE)")
    backtest_results_GARCH = perform_backtesting(stock_returns[1:], np.sqrt(GARCH_skewed_t), dNu_GARCH, dLambda_GARCH, "GARCH (MLE)")
    backtest_results_GJR_GARCH = perform_backtesting(stock_returns[1:], np.sqrt(GJR_GARCH_skewed_t), dNu_GJR_GARCH, dLambda_GJR_GARCH, "GJR-GARCH (MLE)")
    backtest_results_EGARCH = perform_backtesting(stock_returns[1:], np.sqrt(EGARCH_skewed_t), dNu_EGARCH,dLambda_EGARCH, "EGARCH (MLE)")
    backtest_results_GAS = perform_backtesting(stock_returns[1:], np.sqrt(GAS_skewed_t), dNu_GAS,dLambda_GAS, "GAS (MLE)")

    # Perform backtesting for all models (RWMHM)
    backtest_results_ARCH_rwmhm = perform_backtesting(stock_returns[1:], rwmhm_ARCH, dNu_ARCH, dLambda_ARCH, "ARCH (RWMHM)")
    backtest_results_GARCH_rwmhm = perform_backtesting(stock_returns[1:], rwmhm_GARCH, dNu_GARCH, dLambda_GARCH, "GARCH (RWMHM)")
    backtest_results_GJR_GARCH_rwmhm = perform_backtesting(stock_returns[1:], rwmhm_GJR_GARCH, dNu_GJR_GARCH, dLambda_GJR_GARCH, "GJR-GARCH (RWMHM)")
    backtest_results_EGARCH_rwmhm = perform_backtesting(stock_returns[1:], rwmhm_EGARCH, dNu_EGARCH, dLambda_EGARCH, "EGARCH (RWMHM)")
    backtest_results_GAS_rwmhm = perform_backtesting(stock_returns[1:], rwmhm_GAS, dNu_GAS,dLambda_GAS, "GAS (RWMHM)")

    # Print all backtest results
    print("\n=== Backtesting Results ===")
    print_backtest_results(backtest_results_ARCH)
    print_backtest_results(backtest_results_GARCH)
    print_backtest_results(backtest_results_GJR_GARCH)
    print_backtest_results(backtest_results_EGARCH)
    print_backtest_results(backtest_results_GAS)

    print("\n=== Backtesting Results (RWMHM) ===")
    print_backtest_results(backtest_results_ARCH_rwmhm)
    print_backtest_results(backtest_results_GARCH_rwmhm)
    print_backtest_results(backtest_results_GJR_GARCH_rwmhm)
    print_backtest_results(backtest_results_EGARCH_rwmhm)
    print_backtest_results(backtest_results_GAS_rwmhm)
    
    # Perform Diebold-Mariano tests with all loss functions
    print("\n=== Diebold-Mariano Tests ===")
    models_dict_mle = {
        'ARCH': np.sqrt(ARCH_skewed_t),
        'GARCH': np.sqrt(GARCH_skewed_t),
        'GJR-GARCH': np.sqrt(GJR_GARCH_skewed_t),
        'EGARCH': np.sqrt(EGARCH_skewed_t),
        'GAS': np.sqrt(GAS_skewed_t)
        }

    # Compare MLE models using all loss functions
    for loss_func in ['MSE', 'RMSE', 'MAE', 'MAPE']:
        print(f"\nMLE Models Comparison ({loss_func} loss):")
        dm_results = compare_models_with_dm(stock_returns[1:], models_dict_mle, loss_func)
        print(dm_results.to_string(index=False))
    
    # Compare RWMHM models using all loss functions
    models_dict_rwmhm = {
        'ARCH': rwmhm_ARCH,
        'GARCH': rwmhm_GARCH,
        'GJR-GARCH': rwmhm_GJR_GARCH,
        'EGARCH': rwmhm_EGARCH,
        'GAS': rwmhm_GAS
        }

    for loss_func in ['MSE', 'RMSE', 'MAE', 'MAPE']:
        print(f"\nRWMHM Models Comparison ({loss_func} loss):")
        dm_results = compare_models_with_dm(stock_returns[1:], models_dict_rwmhm, loss_func)
        print(dm_results.to_string(index=False))