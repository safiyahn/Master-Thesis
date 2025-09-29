#Names: Alyde Bles, Khanh Van Le, Safiyah Niamat, Floris Hoefsloot
#Student numbers: 2826862, 2676244, 2631412, 2825606

import pandas as pd  # to import and process data
import numpy as np  # to make calculations and arrays
import scipy.stats as st  # for optimization problems
import scipy.optimize   as sc  # for optimization problems
import matplotlib.pyplot as plt  # for making graphs
import math  # to use log gamma function
import numdifftools as nd
from scipy.stats import t
from scipy.stats import norm

df1 = pd.read_csv("/Users/floris/Documents/Climate Econometrics/Advanced Econometrics/Group Assignment/GA_datafiles/spectra_motors_returns.csv", sep = ",", index_col=[0])

df1['date'] = pd.to_datetime(df1['date'], format='%d/%m/%Y')

df2 = pd.read_csv("/Users/floris/Documents/Climate Econometrics/Advanced Econometrics/Group Assignment/GA_datafiles/competitors_returns.csv", sep = ",", 
                  index_col=[0])
df2['date'] = pd.to_datetime(df2['date'], format='%d/%m/%Y')

data = df1.merge(df2, how='outer')
for ticker in data['TICKER'].unique():
  data.loc[data['TICKER']==ticker, 'RET'] = (data.loc[data['TICKER']==ticker, 'RET'] - data.loc[data['TICKER']==ticker, 'RET'].mean())*100


#  Question 1 plots 
def Q1_plots(delta_values , lambda_values, parameters, sigma_lagged):
    # setting the parameters
    omega, alpha, beta = parameters
    x = np.arange(-10,10,0.01)
    # Make a loop to model for different values for delta and lambda
    for d in delta_values:
        plt.figure(figsize = (20,10))
        for l in lambda_values:
            numerator = alpha * (x ** 2) + d * (x ** 2) * (x<0)
            denominator = 1 + (l * (x ** 2)/ sigma_lagged )
            variance = omega + numerator / denominator + beta * sigma_lagged
            plt.plot(x,variance,label =  fr"$\lambda$ ={l} and $\delta$ ={d}")
            plt.ylabel ("volatility", rotation =90,fontsize=20)
            plt.xlabel(r"$x_{t}$",fontsize=20)
            plt.legend()
    
#Question 1 printing and set the values
lambda_values = [0 , 0.01 , 0.1 , 1]
delta_values = [0 , 0.1 , 0.2 , 1]
parameters = [0, 0.05, 0.9]
sigma_lagged =1 
Q1_plots(lambda_values,delta_values, parameters, sigma_lagged)    

#Question 2
# First make the plots for each ticker
def Q2_plots(stock,ticker):
    plt.figure(figsize = (20,10))
    date = pd.to_datetime(stock['date'], format="%d/%m/%Y" )
    plt.plot(date,stock['RET'])
    plt.title(f'The Daily Returns of {ticker}',fontsize=20)
    plt.ylabel('The Returns',fontsize=10)
    plt.xlabel('Years',fontsize=10)

# The descriptive stats table 
def Q2_stats(data, ticker):
    print(f'\nDescriptive Statistics for Company: {ticker}')
    stats = data['RET'].describe()
    print('The Number of Observations:', len(data))
    print(stats[['mean', '50%', 'std', 'min', 'max']])
    print('The Skewness:', st.skew(data['RET']))
    print('The Excess Kurtosis:', st.kurtosis(data['RET']))

# printing
for ticker in data['TICKER'].unique():
    data_ticker = data[data['TICKER'] == ticker].reset_index()
    Q2_plots(data_ticker, ticker)
    Q2_stats(data_ticker, ticker)
    

# QUESTION 3 model (i) non-robust GARCH (lambda = 0, delta = 0)

###FUNCTION##################################


def llik_fun_non_robust_GARCH(theta, x):
    T = len(x)
    omega = theta[0]
    alpha = theta[1]
    beta = theta[2]
    nu = theta[3]

    sig2 = np.zeros(T + 1)
    l = np.zeros(T)
    sig2[0] = np.var(x[0:50])
    indicator = np.zeros(T)
    indicator_2 = []
    indicator_3 = np.zeros(T)
    c = 0.0001
    sub_sum = np.zeros(T)
    sub_sum_result = []
    l_variant = []

    for t in range(0, T):
        if x[t] < 0:
            indicator[t] = 1
        else:
            indicator[t] = 0

        sig2[t + 1] = omega + ((alpha * x[t] ** 2) + beta * sig2[t])

    if min(sig2) < 0 or nu < c:
        for u in range(0, T):
            if sig2[u] < 0:
                indicator_3[u] = 1
            else:
                indicator_3[u] = 0

            sub_sum[u] = (sig2[u] ** 2) * indicator_3[u]
        sub_sum_result = np.sum(sub_sum)

        if nu < c:
            indicator_2 = 1
        else:
            indicator_2 = 0

        l_variant = -100 * T - 10000 * T * ((((nu - c) ** 2) * indicator_2) + sub_sum_result)
        return -l_variant

    else:
        for t in range(0, T):
            l[t] = math.lgamma(((nu + 1) / 2)) \
                   - 0.5 * np.log(nu * np.pi) \
                   - math.lgamma(nu / 2) \
                   - ((nu + 1) / 2) \
                   * np.log(1 + (((x[t] / (np.sqrt(sig2[t]))) ** 2) / nu)) \
                   - 0.5 * np.log(sig2[t])

        # mean log likelihood
        return -np.sum(l)


# QUESTION 3 

def llik_fun_robust_GARCH(theta, x):
    T = len(x)
    omega = theta[0]
    alpha = theta[1]
    beta = theta[2]
    nu = theta[3]
    lambda_value = theta[4]

    sig2 = np.zeros(T + 1)
    l = np.zeros(T)
    sig2[0] = np.var(x[0:50])
    indicator = np.zeros(T)
    indicator_2 = []
    indicator_3 = np.zeros(T)
    c = 0.0001
    sub_sum = np.zeros(T)
    sub_sum_result = []
    l_variant = []

    for t in range(0, T):
        if x[t] < 0:
            indicator[t] = 1
        else:
            indicator[t] = 0

        sig2[t + 1] = omega + ((alpha * x[t] ** 2) / ( 1 + (lambda_value * x[t] ** 2) /  sig2[t]) + beta * sig2[t])

    if min(sig2) < 0 or lambda_value < c:

        for u in range(0, T):
            if sig2[u] < 0:
                indicator_3[u] = 1
            else:
                indicator_3[u] = 0

            sub_sum[u] = (sig2[u] ** 2) * indicator_3[u]
        sub_sum_result = np.sum(sub_sum)

        if lambda_value < c:
            indicator_2 = 1
        else:
            indicator_2 = 0

        l_variant = -100 * T - 10000 * T * ((((lambda_value - c) ** 2) * indicator_2) + sub_sum_result)
        return -l_variant

    else:
        for t in range(0, T):
            l[t] = math.lgamma(((nu + 1) / 2)) \
                   - 0.5 * np.log(nu * np.pi) \
                   - math.lgamma(nu / 2) \
                   - ((nu + 1) / 2) \
                   * np.log(1 + (((x[t] / (np.sqrt(sig2[t]))) ** 2) / nu)) \
                   - 0.5 * np.log(sig2[t])

        # mean log likelihood
        return -np.sum(l)


def llik_fun_robust_GARCH_leverage(theta, x):
    T = len(x)
    omega = theta[0]
    alpha = theta[1]
    beta = theta[2]
    nu = theta[3]
    lambda_value = theta[4]
    delta = theta[5]

    sig2 = np.zeros(T + 1)
    l = np.zeros(T)
    sig2[0] = np.var(x[0:50])
    indicator = np.zeros(T)
    indicator_2 = []
    indicator_3 = np.zeros(T)
    c = 0.0001
    sub_sum = np.zeros(T)
    sub_sum_result = []
    l_variant = []

    for t in range(0, T):
        if x[t] < 0:
            indicator[t] = 1
        else:
            indicator[t] = 0

        sig2[t + 1] = omega + ((alpha * x[t] ** 2 + delta * (x[t] ** 2) * indicator[t]) / (
                1 + (lambda_value * x[t] ** 2) /  sig2[t]) \
                               + beta * sig2[t])

    if min(sig2) < 0 or lambda_value < c:

        for u in range(0, T):
            if sig2[u] < 0:
                indicator_3[u] = 1
            else:
                indicator_3[u] = 0

            sub_sum[u] = (sig2[u] ** 2) * indicator_3[u]
        sub_sum_result = np.sum(sub_sum)

        if lambda_value < c:
            indicator_2 = 1
        else:
            indicator_2 = 0

        l_variant = -100 * T - 10000 * T * ((((lambda_value - c) ** 2) * indicator_2) + sub_sum_result)
        return -l_variant

    else:
        for t in range(0, T):
            l[t] = math.lgamma(((nu + 1) / 2)) \
                   - 0.5 * np.log(nu * np.pi) \
                   - math.lgamma(nu / 2) \
                   - ((nu + 1) / 2) \
                   * np.log(1 + (((x[t] / (np.sqrt(sig2[t]))) ** 2) / nu)) \
                   - 0.5 * np.log(sig2[t])

        # mean log likelihood
        return -np.sum(l)



def initialize_volatility(x):
    return np.var(x[0:50])

def optimize_and_print(model, stock, function, theta_ini, args, options):
    results = sc.minimize(function, theta_ini, args=args, options=options, method = 'Nelder-Mead')
    print(f'Parameter estimates for {model} of {stock}:')
    for idx, param in enumerate(['Omega', 'Alpha', 'Beta', 'Nu', 'Lambda', 'Delta'][:len(theta_ini)]):
        print(f'{param} = {results.x[idx]}')
    print()
    print(f'Log Likelihood for {model} of {stock}:')
    loglik_value = -function(results.x, args[0])
    print(loglik_value)
    print()
    return results, results.x, loglik_value

def news_impact_curves_Q3(theta):
    x = np.arange(-10,10,0.001)
    omega = theta[0]
    alpha = theta[1]
    beta = theta[2]
    nu = theta[3]
    labda = theta[4]
    delta = theta[5]
    sigma_t_1 = 1

    var = omega + (alpha*(x**2) + delta*(x**2) * (x<0)) / (1 + labda * (x**2)/ sigma_t_1 ) + beta*sigma_t_1
    return var, x


print('\nQuestion 3:')
SM_stock = data.loc[data['TICKER'] == 'SM', 'RET'].values[0:2500]
KO_stock = data.loc[data['TICKER'] == 'KO', 'RET'].values[0:2500]
options = {'eps': 1e-09, 'disp': True, 'maxiter': 5000}

# Model i
theta_ini_SM_1 = [initialize_volatility(SM_stock) / 50, 0.02, 0.96, 10]
res_SM_1, theta_SM_1, log_lik_SM_1 = optimize_and_print("Non-robust GARCH", "SM", llik_fun_non_robust_GARCH, theta_ini_SM_1, (SM_stock,), options)
# Compute the Hessian
hessian_func_1 = nd.Hessian(llik_fun_non_robust_GARCH)
SE_SM_1 = np.sqrt(np.diag(np.linalg.inv(hessian_func_1(theta_SM_1, SM_stock))))

# Model ii
theta_ini_SM_2 = [initialize_volatility(SM_stock) / 50, 0.02, 0.96, 10, 0]
res_SM_2, theta_SM_2, log_lik_SM_2 =optimize_and_print("robust GARCH without leverage", "SM", llik_fun_robust_GARCH, theta_ini_SM_2, (SM_stock,), options)


# Model iii
theta_ini_SM_3 = [initialize_volatility(SM_stock) / 50, 0.02, 0.96, 10, 0, 0]
res_SM_3, theta_SM_3, log_lik_SM_3 = optimize_and_print("robust GARCH with leverage", "SM", llik_fun_robust_GARCH_leverage, theta_ini_SM_3, (SM_stock,), options)
hessian_func_3 = nd.Hessian(llik_fun_robust_GARCH_leverage)
SE_SM_3 = np.sqrt(np.diag(np.linalg.inv(hessian_func_3(theta_SM_3, SM_stock))))

# # Model i KO to check
# theta_ini_KO_1 = [initialize_volatility(KO_stock) / 50, 0.02, 0.96, 10]
# res_KO_1, theta_KO_1, log_lik_KO_1 = optimize_and_print("Non-robust GARCH", "KO", llik_fun_non_robust_GARCH, theta_ini_KO_1, (KO_stock,), options)
# SE_KO_1 = np.sqrt(np.diag(np.linalg.inv(hessian_func_1(theta_KO_1, KO_stock))))

# # Model iii to check
# theta_ini_KO_3 = [initialize_volatility(KO_stock) / 50, 0.02, 0.96, 10, 0, 0]
# res_KO_3, theta_KO_3, log_lik_KO_3 = optimize_and_print("robust GARCH with leverage", "KO", llik_fun_robust_GARCH_leverage, theta_ini_KO_3, (KO_stock,), options)
# SE_KO_3 = np.sqrt(np.diag(np.linalg.inv(hessian_func_3(theta_KO_3, KO_stock))))



def plot_question3(theta_SM_1, theta_SM_2, theta_SM_3):
    SM_1 = news_impact_curves_Q3(np.append([theta_SM_1],[0,0]))
    SM_2 = news_impact_curves_Q3(np.append([theta_SM_2], [0]))
    SM_3 = news_impact_curves_Q3(theta_SM_3)
    
    plt.figure(figsize = (16,8))
    plt.plot(SM_1[1], SM_1[0], label = "Non-robust GARCH")
    plt.plot(SM_2[1], SM_2[0], label = "Robust GARCH without leverage")
    plt.plot(SM_3[1], SM_3[0], label = "Robust GARCH with leverage")
    plt.legend()
    plt.xlabel(r"$x_{t}$",fontsize=18)
    plt.ylabel (r"$\sigma^2_{t}$", rotation =5,fontsize=18)
    plt.title("News impact curves of SM")
    plt.show()

plot_question3(theta_SM_1, theta_SM_2, theta_SM_3)


# Question 4 
GM_stock = data.loc[data['TICKER'] == 'GM', 'RET'].values[0:2500]
TSLA_stock = data.loc[data['TICKER'] == 'TSLA', 'RET'].values[0:2500]

# Model i GM
theta_ini_GM_1 = [initialize_volatility(GM_stock) / 50, 0.02, 0.96, 10]
res_GM_1, theta_GM_1, log_lik_GM_1 = optimize_and_print("Non-robust GARCH", "GM", llik_fun_non_robust_GARCH, theta_ini_GM_1, (GM_stock,), options)
SE_GM_1 = np.sqrt(np.diag(np.linalg.inv(hessian_func_1(theta_GM_1, GM_stock))))

# Model iii GM
theta_ini_GM_3 = [initialize_volatility(GM_stock) / 50, 0.02, 0.96, 10, 0, 0]
res_GM_3, theta_GM_3, log_lik_GM_3 = optimize_and_print("Robust GARCH with leverage", "GM", llik_fun_robust_GARCH_leverage, theta_ini_GM_3, (GM_stock,), options)
SE_GM_3 = np.sqrt(np.diag(np.linalg.inv(hessian_func_3(theta_GM_3, GM_stock))))


# Model i TSLA
theta_ini_TSLA_1 = [initialize_volatility(TSLA_stock) / 50, 0.02, 0.96, 10]
res_TSLA_1, theta_TSLA_1, log_lik_TSLA_1 = optimize_and_print("Non-robust GARCH", "TSLA", llik_fun_non_robust_GARCH, theta_ini_TSLA_1, (TSLA_stock,), options)
SE_TSLA_1 = np.sqrt(np.diag(np.linalg.inv(hessian_func_1(theta_TSLA_1, TSLA_stock))))


# Model iii TSLA
theta_ini_TSLA_3 = [initialize_volatility(TSLA_stock) / 50, 0.02, 0.96, 10, 0, 0]
res_TSLA_3, theta_TSLA_3, log_lik_TSLA_3 = optimize_and_print("Robust GARCH with leverage", "TSLA", llik_fun_robust_GARCH_leverage, theta_ini_TSLA_3, (TSLA_stock,), options)
SE_TSLA_3 = np.sqrt(np.diag(np.linalg.inv(hessian_func_3(theta_TSLA_3, TSLA_stock))))

def AIC_BIC(log_lik,n,k):
    aic = -2*log_lik+2*k
    bic = -2*log_lik+2*np.log(n)*k
    return aic, bic


aic_SM_1,bic_SM_1 = AIC_BIC(log_lik_SM_1,2500,len(theta_SM_1))
aic_SM_3, bic_SM_3 = AIC_BIC(log_lik_SM_3,2500,len(theta_SM_3))

aic_GM_1,bic_GM_1 = AIC_BIC(log_lik_GM_1,2500,len(theta_GM_1))
aic_GM_3, bic_GM_3 = AIC_BIC(log_lik_GM_3,2500,len(theta_GM_3))

aic_TSLA_1,bic_TSLA_1 = AIC_BIC(log_lik_TSLA_1,2500,len(theta_TSLA_1))
aic_TSLA_3, bic_TSLA_3 = AIC_BIC(log_lik_TSLA_3,2500,len(theta_TSLA_3))

# Question 5 
# (i) testing the use of a robust filter is necessary 
SM_stock_2 = data.loc[data['TICKER'] == 'SM', 'RET'].values
KO_stock_2 = data.loc[data['TICKER'] == 'KO', 'RET'].values
GM_stock_2 = data.loc[data['TICKER'] == 'GM', 'RET'].values
TSLA_stock_2 = data.loc[data['TICKER'] == 'TSLA', 'RET'].values

theta_ini_SM_2 = [initialize_volatility(SM_stock_2) / 50, 0.02, 0.96, 10, 0] #shoudln't we use the MLE estimators? 
res_SM_2, theta_SM_robust, log_lik_SM_2 = optimize_and_print("robust GARCH without leverage", "SM", llik_fun_robust_GARCH, theta_ini_SM_2, (SM_stock_2,), options)

theta_ini_KO_2 = [initialize_volatility(KO_stock_2) / 50, 0.02, 0.96, 10, 0] #shoudln't we use the MLE estimators? 
res_KO_2, theta_KO_robust, log_lik_KO_2 = optimize_and_print("robust GARCH without leverage", "KO", llik_fun_robust_GARCH, theta_ini_KO_2, (KO_stock_2,), options)

theta_ini_GM_2 = [initialize_volatility(GM_stock_2) / 50, 0.02, 0.96, 10, 0]
res_GM_2, theta_GM_robust, log_lik_GM_2 = optimize_and_print("robust GARCH without leverage", "TSLA", llik_fun_robust_GARCH, theta_ini_GM_2, (GM_stock_2,), options)

theta_ini_TSLA_2 = [initialize_volatility(TSLA_stock_2) / 50, 0.02, 0.96, 10, 0]
res_TSLA_2, theta_TSLA_rboust, log_lik_TSLA_2 = optimize_and_print("robust GARCH without leverage", "TSLA", llik_fun_robust_GARCH, theta_ini_TSLA_2, (TSLA_stock_2,), options)

hessian_func_2 = nd.Hessian(llik_fun_robust_GARCH)
SE_SM_2 = np.sqrt(np.diag(np.linalg.inv(hessian_func_2(theta_SM_robust, SM_stock_2))))
SE_KO_2 = np.sqrt(np.diag(np.linalg.inv(hessian_func_2(theta_KO_robust, KO_stock_2))))
SE_GM_2 = np.sqrt(np.diag(np.linalg.inv(hessian_func_2(theta_GM_robust, GM_stock_2))))
SE_TSLA_2 = np.sqrt(np.diag(np.linalg.inv(hessian_func_2(theta_TSLA_rboust, TSLA_stock_2))))

def t_test_robust(SE, theta, stock, name_stock):
    degrees_of_freedom = len(stock) - len(theta)
    print("degrees of freedom =", degrees_of_freedom)
    critical_value = norm.ppf(1 - 0.05)
    lambda_SE = SE[4]
    estimated_lambda = theta[4]
    t_test = estimated_lambda / lambda_SE
    
    if abs(t_test) > critical_value:
      print("Reject H0 for", name_stock, ": the use of a robust filter is necessary ")
    else:
          print("Fail to reject H0 for", name_stock,": the use of a robust filter is not necessary")
             
t_test_SM2 = t_test_robust(SE_SM_2, theta_SM_robust, SM_stock_2, "SM")
t_test_KO2 = t_test_robust(SE_KO_2, theta_KO_robust, KO_stock_2, "KO")
t_test_GM2 = t_test_robust(SE_GM_2, theta_GM_robust, GM_stock_2, "GM")
t_test_TSLA2 = t_test_robust(SE_TSLA_2, theta_TSLA_rboust, TSLA_stock_2, "TSLA")    


# (ii) testing the presence of the leverage effect 

theta_ini_KO_3 = [initialize_volatility(KO_stock_2) / 50, 0.02, 0.96, 10, 0, 0]
res_KO_3, theta_KO_leverage, log_lik_KO_3 = optimize_and_print("robust GARCH with leverage", "KO", llik_fun_robust_GARCH_leverage, theta_ini_KO_3, (KO_stock_2,), options)
SE_KO_3 = np.sqrt(np.diag(np.linalg.inv(hessian_func_3(theta_KO_leverage, KO_stock))))

theta_ini_SM_3 = [initialize_volatility(SM_stock_2) / 50, 0.02, 0.96, 10, 0, 0] #shoudln't we use the MLE estimators? 
res_SM_3, theta_SM_leverage, log_lik_SM_3 = optimize_and_print("robust GARCH with leverage", "SM", llik_fun_robust_GARCH_leverage, theta_ini_SM_3, (SM_stock_2,), options)
SE_SM_3 = np.sqrt(np.diag(np.linalg.inv(hessian_func_3(theta_SM_leverage, SM_stock_2))))

theta_ini_GM_3 = [initialize_volatility(GM_stock_2) / 50, 0.02, 0.96, 10, 0, 0]
res_GM_3, theta_GM_leverage, log_lik_GM_3 = optimize_and_print("robust GARCH with leverage", "TSLA", llik_fun_robust_GARCH_leverage, theta_ini_GM_3, (GM_stock_2,), options)
SE_GM_3 = np.sqrt(np.diag(np.linalg.inv(hessian_func_3(theta_GM_leverage, GM_stock_2))))

theta_ini_TSLA_3 = [initialize_volatility(TSLA_stock_2) / 50, 0.02, 0.96, 10, 0, 0]
res_TSLA_3, theta_TSLA_leverage, log_lik_TSLA_3 = optimize_and_print("robust GARCH with leverage", "TSLA", llik_fun_robust_GARCH_leverage, theta_ini_TSLA_3, (TSLA_stock_2,), options)
SE_TSLA_3 = np.sqrt(np.diag(np.linalg.inv(hessian_func_3(theta_TSLA_leverage, GM_stock_2))))

def t_test_leverage(SE, theta, stock, name_stock):
    degrees_of_freedom = len(stock) - len(theta)
    print("degrees of freedom =", degrees_of_freedom)
    critical_value = norm.ppf(0.05)
    delta_SE = SE[5]
    estimated_delta = theta[5]
    t_test = estimated_delta / delta_SE
    
    if t_test < -critical_value:
      print("Reject H0 for", name_stock, ": the leverage effect is present")
    else:
          print("Fail to reject H0 for", name_stock,": the leverage effect is not present")
             
t_test_SM3 = t_test_leverage(SE_SM_3, theta_SM_leverage, SM_stock_2, "SM")
t_test_KO3 = t_test_leverage(SE_KO_3, theta_KO_leverage, KO_stock_2, "KO")
t_test_GM3 = t_test_leverage(SE_GM_3, theta_GM_leverage, GM_stock_2, "GM")
t_test_TSLA3 = t_test_leverage(SE_TSLA_3, theta_TSLA_leverage, TSLA_stock_2, "TSLA")


#Question 6
def news_impact_curves_Q6(theta):
    x = np.arange(-10,10,0.01)
    omega = theta[0]
    alpha = theta[1]
    beta = theta[2]
    nu = theta[3]
    labda = theta[4]
    delta = theta[5]
    sigma_t_1 = 1      
    
    var = omega + (alpha*(x**2) + delta*(x**2) * (x<0)) / (1 + labda * (x**2)/sigma_t_1) + beta*sigma_t_1
    return var

def volatility(theta, stock, leverage):
    sigma2 = np.zeros(len(stock))
    sigma2[0] = np.var(stock[:50])

    for i in range(0, len(stock) - 1):
        var_term = theta[1] * stock[i]**2 / (1 + theta[4] * stock[i]**2 / sigma2[i])
        if leverage:
            var_term += theta[5] * stock[i]**2 * int(stock[i] < 0) / (1 + theta[4] * stock[i]**2 / sigma2[i])
        sigma2[i + 1] = theta[0] + var_term + theta[2] * sigma2[i]

    return sigma2

var_SM = news_impact_curves_Q6(np.append(theta_SM_robust, 0))
var_GM = news_impact_curves_Q6(np.append(theta_GM_robust, 0))
var_TSLA = news_impact_curves_Q6(np.append(theta_TSLA_rboust, 0))
var_stocks = [var_SM,var_GM,var_TSLA]

var_SM_lev = news_impact_curves_Q6(theta_SM_leverage)
var_GM_lev = news_impact_curves_Q6(theta_GM_leverage)
var_TSLA_lev = news_impact_curves_Q6(theta_TSLA_leverage)
var_stocks_leverage = [var_SM_lev,var_GM_lev,var_TSLA_lev]

stocks = ['SM','GM', 'TSLA']

fig, axs= plt.subplots(3,2, figsize = (12,12))

for i in range(len(stocks)):
    time = np.arange(-10, 10, 0.01)
    axs[i,0].plot(time, var_stocks[i], linestyle='dashed', color='blue', label='Variance of Stocks')
    axs[i,0].plot(time, var_stocks_leverage[i], linestyle='dashed', color='red', label='Variance of Stocks with Leverage')
    axs[i,0].set_xlabel('X_t')
    axs[i,0].set_ylabel('Sigma_t')
    axs[i,0].set_title(f'News impact curve stock {stocks[i]}')
    axs[i,0].legend()

    #Adding space between subplots
    if i < len(stocks) - 1:
        axs[i,0].set_xticklabels([])
        axs[i,0].set_xlabel('')


sigma2_SM = volatility(np.append(theta_SM_robust, 0), SM_stock_2, False)
sigma2_SM_leverage = volatility(theta_SM_leverage, SM_stock_2, True)

sigma2_GM = volatility(np.append(theta_GM_robust, 0), GM_stock_2, False)
sigma2_GM_leverage = volatility(theta_GM_leverage, GM_stock_2, True)

sigma2_TSLA = volatility(np.append(theta_TSLA_rboust, 0), TSLA_stock_2, False)
sigma2_TSLA_leverage = volatility(theta_TSLA_leverage, TSLA_stock_2, True)

sigma2_combined = [sigma2_SM, sigma2_GM, sigma2_TSLA]
sigma2_combined_leverage = [sigma2_SM_leverage, sigma2_GM_leverage, sigma2_TSLA_leverage]

for i in range(len(stocks)):
    time = np.arange(len(sigma2_combined[i]))
    axs[i,1].plot(time, sigma2_combined[i], linewidth=0.5, label='Without Leverage', color='blue')
    axs[i,1].plot(time, sigma2_combined_leverage[i], linewidth=0.5, label='With Leverage', color='green')
    axs[i,1].axvline(2500, color='red', linestyle='--')

    axs[i,1].set_title(f'Filtered volatilities of {stocks[i]}')
    axs[i,1].set_xlabel('Time')
    axs[i,1].set_ylabel('Returns')
    axs[i,1].legend()

    # Add some space between subplots
    if i < len(stocks) - 1:
        axs[i,1].set_xticklabels([])
        axs[i,1].set_xlabel('')
    
#Question 7
def volatility2(theta, stock, leverage=False):
    sigma2 = np.zeros(len(stock))
    sigma2[0] = np.var(stock[:50])

    for i in range(0, len(stock) - 1):
        var_term = theta[1] * stock[i]**2 + theta[2] * sigma2[i]

        if leverage:
            var_term += theta[5] * stock[i]**2 * int(stock[i] < 0) / (1 + theta[4] * stock[i]**2 / sigma2[i])

        sigma2[i + 1] = theta[0] + var_term

    return sigma2

def simReturns(stock, sigma2_Q7, theta, horizon_level, leverage=False):
    sim = 1000
    
    horizon_levels = [1, 5, 20]
    quantiles = [0.01, 0.05, 0.1]
    
    results = {level: [] for level in horizon_levels}

    for j in range(sim):
        x = [stock]
        sigma2 = [sigma2_Q7]

        for i in range(horizon_level):
            var_term = theta[1] * x[i]**2 + theta[2] * sigma2[i]

            if leverage:
                var_term += theta[5] * x[i]**2 * int(x[i] < 0) / (1 + theta[4] * x[i]**2 / sigma2[i])

            sigma2.append(theta[0] + var_term)
            x.append(np.sqrt(sigma2[i + 1]) * np.random.standard_t(theta[3]))

        x = x[1:]

        compound = (np.array(x) / 100) + 1
        for level in horizon_levels:
            results[level].append(100 * (np.prod(compound[:level]) - 1))

    for level in horizon_levels:
        quantile_results = np.quantile(results[level], quantiles)
        print(f"Horizon {level} Quantiles:", quantile_results)
    print()

def VaR_estimate(theta, stock, leverage):
    date = '2021-01-04'
    sigma2 = volatility2(theta, stock.loc[:date], leverage)

    horizon_level = 20
    simReturns(stock[date], sigma2[-1], theta, horizon_level, leverage)    
    
    
data = data.copy()
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
data = data.set_index('date')
   
#CHECK FOR KO
theta_ini_KO_1 = [initialize_volatility(KO_stock) / 50, 0.02, 0.96, 10]
res_KO_1, theta_KO_1, log_lik_KO_1 =optimize_and_print("Non-robust GARCH", "KO", llik_fun_non_robust_GARCH, theta_ini_KO_1, (KO_stock,), options)

theta_ini_KO_3 = [initialize_volatility(KO_stock) / 50, 0.02, 0.96, 10, 0, 0]
res_KO_3, theta_KO_3, log_lik_KO_3 = optimize_and_print("robust GARCH with leverage", "KO", llik_fun_robust_GARCH_leverage, theta_ini_KO_3, (KO_stock,), options)

print('VaR values for: KO')
stock_KO = data[data['TICKER'] == 'KO']['RET']
print('Non-robust and No leverage:')
VaR_estimate(theta_KO_1, stock_KO, False)  #without leverage
print('Robust with leverage:')
VaR_estimate (theta_KO_3, stock_KO, True) #with leverage

theta_ini_GM_1 = [initialize_volatility(GM_stock) / 50, 0.02, 0.96, 10]
res_GM_1, theta_GM_1, log_lik_GM_1 =optimize_and_print("robust GARCH without leverage", "GM", llik_fun_non_robust_GARCH, theta_ini_GM_1, (GM_stock,), options)

theta_ini_TSLA_1 = [initialize_volatility(TSLA_stock) / 50, 0.02, 0.96, 10]
res_TSLA_1, theta_TSLA_1, log_lik_TSLA_1 =optimize_and_print("robust GARCH without leverage", "TSLA", llik_fun_non_robust_GARCH, theta_ini_TSLA_1, (TSLA_stock,), options)

print('VaR values for: SM')
stock_SM = data[data['TICKER'] == 'SM']['RET']
print('WITHOUT leverage:')
VaR_estimate(theta_SM_1, stock_SM, False)  #without leverage
print('leverage:')
VaR_estimate (theta_SM_3, stock_SM, True) #with leverage

print('VaR values for: GM')
stock_GM = data[data['TICKER'] == 'GM']['RET']
print('WITHOUT leverage:')
VaR_estimate(theta_GM_1, stock_GM, False)  #without leverage
print('leverage:')
VaR_estimate (theta_GM_3, stock_GM, True) #with leverage

print('VaR values for: TSLA')
stock_TSLA = data[data['TICKER'] == 'TSLA']['RET']
print('WITHOUT leverage:')
VaR_estimate(theta_TSLA_1, stock_TSLA, False)  #without leverage
print('leverage:')
VaR_estimate (theta_TSLA_3, stock_TSLA, True) #with leverage


#Question 8
# First calculating the the VAR 
def forecasting_VaR(stock, theta, a, leverage):
    returns = stock[2500:]
    sigma_fc = volatility2(theta, stock, leverage)[2500:]
    VaR = np.sqrt(sigma_fc) * st.t.ppf(a, theta[3])
    return returns, VaR

def calculated_SE_NW(returns, violations):
    High = len(returns)
    Low = int(High **(1/5))
    y = violations - np.mean(violations)
    first_summation = np.sum(y ** 2)
    second_summation = 0
    
    for h in range(Low, High):
        w = 1 - (h + 1) / (Low + 1)
    for l in range(Low):
        second_summation += w * y[h] * y[h - l - 1]

    standard_error_NW = np.sqrt(1/High*first_summation+2/High*second_summation)/np.sqrt(High)
    
    return standard_error_NW

def calculate_hit_rate(returns, VaR, a):
    N = len(returns)
    violations = np.where(returns < VaR , 1 , 0)
    hit_rate = round(np.sum(violations)/N,3)
    standard_error = round(np.std ((violations)/np. sqrt (N)),3)
    standard_error_NW_a = calculated_SE_NW(returns , violations)
    print (f"Hit rate: {hit_rate}")    
    print (f"Standard error: {standard_error}")    
    print (f"Mis-specification robust standard error: {round(standard_error_NW_a,3) } \n")
    

for a in [0.01, 0.05, 0.1]:
    print(f'VAR({a}):')
    stock = data[data['TICKER'] == "KO"]['RET']
    print('Non-robust and No leverage KO:')
    returns, VaR_no_leverage = forecasting_VaR(stock, theta_KO_1, a, False)
    calculate_hit_rate(returns , VaR_no_leverage,a)
    
    #with leverage:
    print('Robust with leverage KO:')
    returns, VaR_leverage = forecasting_VaR(stock, theta_KO_3, a, True)
    calculate_hit_rate(returns, VaR_leverage, a)
        
        
for a in [0.01, 0.05, 0.1]:
    print(f'VAR({a}):')
    stock = data[data['TICKER'] == 'SM']['RET']
    print('Non-robust and No leverage SM:')
    returns, VaR_no_leverage = forecasting_VaR(stock, theta_SM_1, a, False)
    calculate_hit_rate(returns , VaR_no_leverage,a)
    
    #with leverage:
    print('Robust with leverage SM:')
    returns, VaR_leverage = forecasting_VaR(stock, theta_SM_3, a, True)
    calculate_hit_rate(returns, VaR_leverage, a)
 
        
for a in [0.01, 0.05, 0.1]:
    print(f'VAR({a}):')
    stock = data[data['TICKER'] == 'GM']['RET']
    print('Non-robust and No leverage GM:')
    returns, VaR_no_leverage = forecasting_VaR(stock, theta_GM_1, a, False)
    calculate_hit_rate(returns , VaR_no_leverage,a)
    
    #with leverage:
    print('Robust with leverage GM:')
    returns, VaR_leverage = forecasting_VaR(stock, theta_GM_3, a, True)
    calculate_hit_rate(returns, VaR_leverage, a)    
        
        
for a in [0.01, 0.05, 0.1]:
    print(f'VAR({a}):')
    stock = data[data['TICKER'] == 'TSLA']['RET']
    print('Non-robust and No leverage TSLA:')
    returns, VaR_no_leverage = forecasting_VaR(stock, theta_TSLA_1, a, False)
    calculate_hit_rate(returns , VaR_no_leverage,a)
    
    #with leverage:
    print('Robust with leverage TSLA:')
    returns, VaR_leverage = forecasting_VaR(stock, theta_TSLA_3, a, True)
    calculate_hit_rate(returns, VaR_leverage, a)