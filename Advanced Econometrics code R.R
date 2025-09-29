#Names: Alyde Bles, Khanh Van Le, Safiyah Niamat, Floris Hoefsloot
#Student numbers: 2826862, 2676244, 2631412, 2825606

#Setting working directory and loading data (group 37)----
setwd(
  "/Users/floris/Documents/Climate Econometrics/Advanced Econometrics/Group Assignment"
)

Data_QA <-
  read.csv("GA_datafiles/GA_datafiles_Quality_Assurance/components_37.csv")
Data_Marketing <-
  read.csv("GA_datafiles/GA_datafiles_Marketing_and_Sales/marketing_data.csv")
Data_Pricing <-
  read.csv("GA_datafiles/GA_datafiles_Marketing_and_Sales/pricing_data.csv")

#Install packages and load libraries----
#install.packages("ggplot2")
#install.packages("dplyr")
#install.packages("numDeriv")
#install.packages("ivreg")

#Load libraries
library(ggplot2)
library(dplyr)
library(numDeriv)
library(ivreg)

#Descriptive statistics and plots----
table(Data_QA$Component) #proportions of the different components that failed and were replaced
length(unique(Data_QA$Vehicle.ID)) #number of vehicles

#Boxplot of lifetime of different components
casket_data <- Data_QA[Data_QA$Component == "casket",]
hinge_data <- Data_QA[Data_QA$Component == "hinge",]
buckle_data <- Data_QA[Data_QA$Component == "buckle",]
tire_data <- Data_QA[Data_QA$Component == "tire",]
battery_data <- Data_QA[Data_QA$Component == "battery",]
boxplot(
  casket_data$FTD,
  hinge_data$FTD,
  buckle_data$FTD,
  tire_data$FTD,
  battery_data$FTD,
  names = c("Casket", "Hinge", "Buckle", "Tire", "Battery"),
  main = "Lifetime of Components"
)
mtext("Years", side = 2, line = 3)



#Question 1.1----
#Creating function for the calculation of the log likelihood
Q1.1_func <- function(beta_0, beta_1, beta_2, data) {
  lambda <- exp(beta_0 + beta_1 * data$RQ + beta_2 * data$ATL)
  
  #Initialising vector for log likelihood
  log_likelihood <- rep(NA, nrow(data))
  
  #For loop for calculation of log-likelihood
  for (t in 1:nrow(data)) {
    if (data$Fail.time[t] == "31/12/2021") {
      log_likelihood[t] = log(exp(-lambda[t] * data$FTD[t]))
    } else {
      log_likelihood[t] = log(lambda[t] * exp(-lambda[t] * data$FTD[t]))
    }
  }
  
  return(log_likelihood) #Return the log-likelihood for usage in Question 1.2
}

#Initiate parameters
beta_0 = 1
beta_1 = -0.1
beta_2 = -0.5
Q1.1_answer <-
  sum(Q1.1_func(beta_0, beta_1, beta_2, Data_QA)) / nrow(Data_QA) #Summing the log likelihoods and calculating average
print(paste("Average log likelihood Q1:", Q1.1_answer))

#Question 1.2----
#Function for optimising
Q1.2_func <- function(data, component) {
  #Retrieving corresponding component data from data set
  component_data <- data[data$Component == component, ]
  
  #Defining objective function for optimisation
  objective_function <- function(parameters) {
    beta_0 <- parameters[1]
    beta_1 <- parameters[2]
    beta_2 <- parameters[3]
    return(sum(Q1.1_func(
      beta_0, beta_1, beta_2, component_data
    )))
  }
  
  #Initialising parameters
  beta_0 <- -log(mean(component_data$FTD))
  beta_1 <- 0
  beta_2 <- 0
  par <- c(beta_0, beta_1, beta_2)
  
  #Optimisation of objective function
  optimisation <-
    optim(par,
          objective_function,
          control = list(fnscale = -1),
          hessian = TRUE)
  
  return(optimisation)
}

#Creating vector of components strings
components <- c("casket", "hinge", "buckle", "tire", "battery")
Q1.2_answer <- list()
#Loop through the components and call the Q1.2 function for each one
for (i in components) {
  Q1.2_answer[[i]] <- Q1.2_func(Data_QA, i)
  
  #Print the parameters values for current component
  print(paste("Parameter values for", i))
  print(paste("Beta_0    ", "Beta_1    ", "Beta_2"))
  print(Q1.2_answer[[i]]$par)
  print(paste("Maximized total log likelihood:", Q1.2_answer[[i]]$value))
  cat("\n")
}

#Question 1.2 - Bonus Question----

#Function for calculation of OPG matrix
OPG_calculation <- function(data, component) {
  component_data <-
    data[data$Component == component, ] #Retrieving corresponding component data from data set
  beta_estimates <-
    Q1.2_answer[[component]]$par #Retrieving beta estimates
  
  #Function for the calculation of a single log-likelihood, instead of vector as in Q1.1_func
  calculate_log_likelihood <- function(params, observation) {
    beta_0 <- params[1]
    beta_1 <- params[2]
    beta_2 <- params[3]
    
    lambda <-
      exp(beta_0 + beta_1 * observation$RQ + beta_2 * observation$ATL) #Calculating lambda directly
    
    #Indicator function for the two cases
    if (observation$Fail.time == "31/12/2021") {
      log_likelihood <- log(exp(-lambda * observation$FTD))
    } else {
      log_likelihood <- log(lambda * exp(-lambda * observation$FTD))
    }
    return(log_likelihood)
  }
  
  #Initiating OPG matrix
  Matrix_OPG <- matrix(0, nrow = 3, ncol = 3)
  
  #Calculating gradient and OPG matrix
  for (t in 1:nrow(component_data)) {
    observation <- component_data[t, ]
    g <-
      grad(func = calculate_log_likelihood,
           x = beta_estimates,
           observation = observation)
    Matrix_OPG <- Matrix_OPG + outer(g, g)
  }
  
  return(Matrix_OPG)
  
}

#Calculate Matrices
Matrix_Hessian <- Q1.2_answer[["tire"]]$hessian #Hessian
Matrix_OPG <- OPG_calculation(Data_QA, "tire") #OPG
Matrix_Sandwich <-
  solve(Matrix_Hessian) %*% Matrix_OPG %*% solve(Matrix_Hessian) #Sandwich

#Calculate se's for different matrices
SE_Hessian <- sqrt(diag(solve(-Matrix_Hessian))) #Hessian
SE_OPG <- sqrt(diag(solve(Matrix_OPG))) #OPG
SE_Sandwich <- sqrt(diag((Matrix_Sandwich))) #Sandwich

print(paste(
  "Standard errors for Hessian:",
  round(SE_Hessian[1], 4),
  round(SE_Hessian[2], 4),
  round(SE_Hessian[3], 4)
))
print(paste(
  "Standard errors for OPG:",
  round(SE_OPG[1], 4),
  round(SE_OPG[2], 4),
  round(SE_OPG[3], 4)
))
print(paste(
  "Standard errors for Sandwich:",
  round(SE_Sandwich[1], 4),
  round(SE_Sandwich[2], 4),
  round(SE_Sandwich[3], 4)
))

#Question 1.3----
Q1.3_func <- function(parameters, data, component) {
  #Retrieving corresponding component data from data set
  data <- data[data$Component == component, ]
  
  #Setting parameters
  beta_1 <- parameters[1]
  beta_2 <- parameters[2]
  gamma_0 <- parameters[3]
  gamma_1 <- parameters[4]
  gamma_2 <- parameters[5]
  
  #Initialising vectors
  beta_0 <- rep(NA, nrow(data))
  lambda <- rep(NA, nrow(data))
  log_likelihood <- rep(NA, nrow(data))
  
  #Initialisation of beta0
  beta_0[1] <- gamma_0
  
  #For loop for calculation of log likelihood
  for (i in 1:nrow(data)) {
    lambda[i] <-
      exp(beta_0[i] + (beta_1 * data$RQ[i]) + (beta_2 * data$ATL[i]))
    log_likelihood[i] = Q1.1_func(beta_0[i], beta_1, beta_2, data[i, ]) #Use function from question 1
    
    #Calculate different values for beta_0 depending on FTD
    if (data$Fail.time[i] == "31/12/2021") {
      beta_0[i + 1] <-
        gamma_0 * (1 - gamma_1) + (gamma_1 * beta_0[i]) + gamma_2 * (-lambda[i] * data$FTD[i])
    }
    else{
      beta_0[i + 1] <-
        gamma_0 * (1 - gamma_1) + (gamma_1 * beta_0[i]) +  gamma_2 * (1 - lambda[i] * data$FTD[i])
    }
  }
  return(log_likelihood)
}

#Initiating parameters
parameters <- c(-0.1, -0.5, 1, 0.8, 0.025)

#Calculating correct log likelihood by summing and dividing for the average
Q1.3_answer <-
  sum(Q1.3_func(parameters, Data_QA, "tire")) / nrow(subset(Data_QA, Component == "tire"))
print(paste("Average log likelihood Q3:", Q1.3_answer))

#Question 1.4----
Q1.4_func <- function(data, component) {
  #Retrieving corresponding component data from data set
  data <- data[data$Component == component, ]
  
  #Defining objective function for optimisation
  objective_function <- function(parameters) {
    return(sum(Q1.3_func(parameters, data, component)))
  }
  
  #Initiating parameters
  beta_1 <- Q1.2_answer[[component]]$par[2]
  beta_2 <- Q1.2_answer[[component]]$par[3]
  gamma_0 <- Q1.2_answer[[component]]$par[1]
  gamma_1 <- 1
  gamma_2 <- 0.025
  parameters <- c(beta_1, beta_2, gamma_0, gamma_1, gamma_2)
  
  optimisation <-
    optim(parameters, objective_function, control = list(fnscale = -1))
  return(optimisation)
}

components <-
  c("casket", "hinge", "buckle", "tire", "battery") #Defining vector of the components
Q1.4_answer <- list()
#Loop through the components and call the Q1.4_func for each one
for (i in components) {
  Q1.4_answer[[i]] <- Q1.4_func(Data_QA, i)
  
  #Print the parameters values for current component
  print(paste("Parameter values for", i))
  print(paste("Beta 1    ", "Beta 2    ", "Gamma 0    ", "Gamma 1    ", "Gamma 2    "))
  print(Q1.4_answer[[i]]$par)
  print(paste("Maximized total log likelihood:", Q1.4_answer[[i]]$value))
  cat("\n")
}

#Question 1.5----
Q1.5_func <- function(data, component) {
  #Retrieving corresponding component data from data set
  data <- data[data$Component == component, ]
  
  #Create empty vectors
  beta_0 <- numeric(nrow(data))
  lambda <- numeric(nrow(data))
  
  #Setting parameters equal to answers of Q4
  beta_1 <- Q1.4_answer[[component]]$par[1]
  beta_2 <- Q1.4_answer[[component]]$par[2]
  gamma_0 <- Q1.4_answer[[component]]$par[3]
  gamma_1 <- Q1.4_answer[[component]]$par[4]
  gamma_2 <- Q1.4_answer[[component]]$par[5]
  
  #Initialising beta_0
  beta_0[1] <- gamma_0
  
  #For loop for calculating beta_0_hat
  for (i in 2:(nrow(data))) {
    lambda[i - 1] <-
      exp((beta_0[i - 1]) + beta_1 * data$RQ[i - 1] + beta_2 * data$ATL[i - 1])
    
    if (data$Fail.time[i - 1] == "31/12/2021") {
      beta_0[i] <-
        gamma_0 * (1 - gamma_1) + gamma_1 * beta_0[i - 1] + gamma_2 * (-lambda[i - 1] * data$FTD[i - 1])
    } else {
      beta_0[i] <-
        gamma_0 * (1 - gamma_1) + (gamma_1 * beta_0[i - 1]) +  gamma_2 * (1 - lambda[i - 1] * data$FTD[i - 1])
    }
    
  }
  
  #Retrieve beta_0 from Q2
  beta_0_Q2 <- Q1.2_answer[[component]]$par[1]
  
  #Plot beta_0 and beta0 from Q2
  plot <-
    ggplot(data, aes(x = as.Date(Start.time, format = "%d/%m/%Y"), y = beta_0)) +
    geom_line(color = "red", lty = "dashed") +
    theme_minimal() +
    labs(x = "Start Time",
         y = "Values",
         title = paste("Beta_0 hat", component)) +
    theme(plot.title = element_text(
      hjust = 0.5,
      size = 20,
      face = "bold"
    )) + geom_hline(yintercept =
                      beta_0_Q2)
  
  return(plot)
}

#Create vector for components and list for plots components
components <- c("casket", "hinge", "buckle", "tire", "battery")
Q1.5_answer <- list()

#For loop for plotting all components
for (component in components) {
  Q1.5_answer[[component]] <- Q1.5_func(Data_QA, component)
  print(Q1.5_answer[[component]])
}

#Question 3.1----
#Function for overlapping plot with all three data in same plot
Q3.1_func_overlap <- function(data) {
  #Setting colours for the plot
  colours <- c(
    "Sales" = "blue",
    "Google Expenditures" = "red",
    "YouTube Expenditures" = "yellow"
  )
  
  plot <- ggplot(data, aes(x = 1:nrow(data))) +
    geom_line(aes(y = s, color = "Sales")) +
    geom_line(aes(y = g, color = "Google Expenditures")) +
    geom_line(aes(y = y, color = "YouTube Expenditures")) +
    labs(x = "Hours", y = "Amount (in Euros)") +
    scale_color_manual(values = colours, name = "Legend") +
    guides(color = guide_legend(title = "Legend"))
  
  return(plot)
}

Q3.1_func_overlap(Data_Marketing)

#Function for separate plots with all three data in separate plot
Q3.1_func_seperate <- function(data) {
  plot_sales <- ggplot(data, aes(x = 1:nrow(data), y = s)) +
    geom_line(color = "blue") +
    labs(x = "Hours", y = "Amount (in Euros)") +
    ggtitle("Sales")
  
  plot_google_expenditures <-
    ggplot(data, aes(x = 1:nrow(data), y = g)) +
    geom_line(color = "red") +
    labs(x = "Hours", y = "Amount (in Euros)") +
    ggtitle("Google Expenditures")
  
  plot_youtube_expenditures <-
    ggplot(data, aes(x = 1:nrow(data), y = y)) +
    geom_line(color = "yellow") +
    labs(x = "Hours", y = "Amount (in Euros)") +
    ggtitle("YouTube Expenditures")
  
  return(list(
    plot_sales,
    plot_google_expenditures,
    plot_youtube_expenditures
  ))
}

Q3.1_func_seperate(Data_Marketing)

#Question 3.2----
#Defining nonlinear dynamic parameter model
Q3.2_func <- function(theta, data) {
  mu <- theta[1]
  phi_1 <- theta[2]
  phi_2 <- theta[3]
  delta_1 <- theta[4]
  delta_2 <- theta[5]
  alpha_1 <- theta[6]
  alpha_2 <- theta[7]
  beta_1 <- theta[8]
  beta_2 <- theta[9]
  
  #Initiating length and vectors for storage of estimates
  T <- nrow(data)
  s_hat <- rep(NA, T)
  g_ads <- rep(NA, T)
  y_ads <- rep(NA, T)
  least_squares <- numeric(T)
  
  #Initialising model. Assuming that g_ads[0] and y_ads[0] are equal to 0.
  g_ads[1] <-
    (alpha_1 * mean(data$g)) / (1 - beta_1) #Starting value is unconditional mean
  if (is.infinite(g_ads[1])) {
    g_ads[1] <- 0
  }
  y_ads[1] <-
    (alpha_2 * mean(data$y)) / (1 - beta_2) #Starting value is unconditional mean
  if (is.infinite(y_ads[1])) {
    y_ads[1] <- 0
  }
  s_hat[1] <-
    mu + phi_1 * g_ads[1] ^ delta_1 + phi_2 * y_ads[1] ^ delta_2
  
  #Calculating estimated values
  for (t in 2:T) {
    g_ads[t] <- beta_1 * g_ads[t - 1] + alpha_1 * data$g[t]
    y_ads[t] <- beta_2 * y_ads[t - 1] + alpha_2 * data$y[t]
    s_hat[t] <-
      mu + phi_1 * g_ads[t] ^ delta_1 + phi_2 * y_ads[t] ^ delta_2
    least_squares[t] <- (data$s[t] - s_hat[t]) ^ 2
  }
  
  #Calculate sum of squared residuals
  ssq <- sum((data$s - s_hat) ^ 2)
  return(1 / T * sum(least_squares))
}

#Defining bounds for each parameter (mu, phi_1, phi_2, delta_1, delta_2, alpha_1, alpha_2, beta_1, beta_2)
lower_bounds <- c(0, 0, 0, 0, 0, 0, 0, 0, 0)
upper_bounds <- c(Inf, Inf, Inf, 1, 1, Inf, Inf, 1, 1)

#Initialising parameter values
initial_theta <- c(1, 1, 1, 0.5, 0.5, 5, 5 , 0.9, 0.9)

Q3.2_answer <-
  optim(
    par = initial_theta,
    fn = Q3.2_func,
    data = Data_Marketing,
    method = "L-BFGS-B",
    lower = lower_bounds,
    upper = upper_bounds
  )
cat(
  paste(
    "Parameter values are: ",
    "\n",
    "mu",
    "phi_1",
    "phi_2",
    "delta_1",
    "delta_2",
    "alpha_1",
    "alpha_2",
    "beta_1",
    "beta_2"
  ),
  "\n",
  Q3.2_answer$par,
  "\n"
)
print(paste("Value of the minimised objective function: ", Q3.2_answer$value))

#Question 3.3----
#Calculation of g_ads and y_ads with obtained parameters
T <- nrow(Data_Marketing)
g_ads <- rep(NA, T)
y_ads <- rep(NA, T)

#Retrieving alpha's and beta's from Q3.2
alpha_1_hat <- Q3.2_answer$par[6]
alpha_2_hat <- Q3.2_answer$par[7]
beta_1_hat <- Q3.2_answer$par[8]
beta_2_hat <- Q3.2_answer$par[9]

#Initialise ads at t = 0
g_ads[1] <- alpha_1_hat * Data_Marketing$g[1]
y_ads[1] <- alpha_2_hat * Data_Marketing$y[1]

#For loop for calculating of adstock
for (t in 2:T) {
  g_ads[t] <-
    beta_1_hat * g_ads[t - 1] + alpha_1_hat * Data_Marketing$g[t]
  y_ads[t] <-
    beta_2_hat * y_ads[t - 1] + alpha_2_hat * Data_Marketing$y[t]
}

#Creating time vector from 2 to T
time_vector <- 2:T

#Plot g_ads
plot(
  time_vector,
  g_ads[2:T],
  type = "l",
  col = "blue",
  xlab = "Time (hours)",
  ylab = "Amount (Euros)"
)
lines(time_vector, g_ads[2:T], col = "blue")
title(main = "Filtered adstock Google")

#Plot y_ads
plot(
  time_vector,
  y_ads[2:T],
  type = "l",
  col = "blue",
  xlab = "Time (hours)",
  ylab = "Amount (Euros)"
)
lines(time_vector, y_ads[2:T], col = "blue")
title(main = "Filtered adstock Youtube")

#Question 3.4----
#Defining function to calculate IRF
Q3.4_func <- function(theta, data, shock, shock_time) {
  mu <- theta[1]
  phi_1 <- theta[2]
  phi_2 <- theta[3]
  delta_1 <- theta[4]
  delta_2 <- theta[5]
  alpha_1 <- theta[6]
  alpha_2 <- theta[7]
  beta_1 <- theta[8]
  beta_2 <- theta[9]
  
  #Initiating length and setting values to origin
  T <- nrow(data)
  g_ads <- rep(0, T)
  y_ads <- rep(0, T)
  s_hat <- rep(mu, T)
  
  #Calculating values at shocktime
  g_ads[shock_time] <-
    0 #beta_1 * g_ads[shock_time - 1] + alpha_1 * data$g[t]
  y_ads[shock_time] <-
    beta_2 * y_ads[shock_time - 1] + alpha_2 * shock
  s_hat[shock_time] <-
    mu + phi_1 * g_ads[shock_time] ^ delta_1 + phi_2 * y_ads[shock_time] ^ delta_2
  
  #Calculate values after shocktime, as normal
  n = 1
  for (t in (shock_time + 1):T) {
    g_ads[t] <- 0 #beta_1 * g_ads[t - 1] + alpha_1 * data$g[t]
    y_ads[t] <-
      beta_2 ^ n * y_ads[shock_time] #+ alpha_2 * data$y[t]
    s_hat[t] <-
      mu + phi_1 * g_ads[t] ^ delta_1 + phi_2 * y_ads[t] ^ delta_2
    n = n + 1
  }
  
  return(list(y_ads, s_hat))
}

#Calling function
shock <- 100
shock_time <- 50
IRF <-
  as.data.frame(Q3.4_func(Q3.2_answer$par, Data_Marketing, shock, shock_time))
IRF <- na.omit(IRF)
colnames(IRF) <- c("y_ads", "s_hat")

#Plotting IRF
plot(
  IRF[, 1],
  col = "blue",
  xlab = "Time (hours)",
  ylab = "Amount (Euros)",
  type = "l",
  xlim = c(0, 200),
  ylim = c(0, max(IRF$y_ads, na.rm = TRUE))
)
lines(IRF[, 2], col = "red")  #s_hat
title(main = "Impulse Response Function for YouTube Marketing")
legend(
  "topright",
  legend = c("y_ads", "s_hat"),
  col = c("blue", "red"),
  lty = 1
)

#Question 3.6
#Accumulated additional sales
shock_time <- 50 #Same as shock_time in Q.3.4
shock <- 0 #No shock instead of 0
sales <-
  as.data.frame(Q3.4_func(Q3.2_answer$par, Data_Marketing, shock, shock_time))
colnames(sales) <- c("y_ads", "s_hat")

#Calculating accumulated additional sales
acc_add_sales <- sum(IRF$s_hat) - sum(sales$s_hat)
print(acc_add_sales)

#Question 3.7----
shock <- 300
shock_time <- 50
IRF_3_7 <-
  as.data.frame(Q3.4_func(Q3.2_answer$par, Data_Marketing, shock, shock_time))
IRF_3_7 <- na.omit(IRF_3_7)
colnames(IRF_3_7) <- c("y_ads", "s_hat")

#Plotting IRF_3_7
plot(
  IRF_3_7[, 1],
  col = "blue",
  xlab = "Time (hours)",
  ylab = "Amount (Euros)",
  type = "l",
  xlim = c(0, 200),
  ylim = c(0, max(IRF_3_7$y_ads, na.rm = TRUE))
)
lines(IRF_3_7[, 2], col = "red")  #s_hat
title(main = "Impulse Response Function for YouTube Marketing")
legend(
  "topright",
  legend = c("y_ads", "s_hat"),
  col = c("blue", "red"),
  lty = 1
)

#Calculating accumulated additional sales
shock_time <- 50 #Same as shock_time in Q.3.4
shock <- 0
sales_3_7 <-
  as.data.frame(Q3.4_func(Q3.2_answer$par, Data_Marketing, shock, shock_time))
colnames(sales_3_7) <- c("y_ads", "s_hat")

#Accumulated additional sales
acc_add_sales_3_7 <- sum(IRF_3_7$s_hat) - sum(sales_3_7$s_hat)
print(acc_add_sales_3_7)

#Question 3.9----
#Function for plotting data
Q3.9_func <- function(data) {
  plot_sales <- ggplot(data, aes(x = 1:nrow(data), y = s)) +
    geom_line(color = "blue") +
    labs(x = "Days", y = "Amount (in Euros)") +
    ggtitle("Sales") +
    theme(plot.title = element_text(hjust = 0.5))
  
  plot_prices <- ggplot(data, aes(x = 1:nrow(data), y = p)) +
    geom_line(color = "red") +
    labs(x = "Days", y = "Amount (in Euros)") +
    ggtitle("Prices") +
    theme(plot.title = element_text(hjust = 0.5))
  
  plot_acquisition_costs <-
    ggplot(data, aes(x = 1:nrow(data), y = c)) +
    geom_line(color = "black") +
    labs(x = "Days", y = "Amount (in Euros)") +
    ggtitle("Acquisition Costs") +
    theme(plot.title = element_text(hjust = 0.5))
  
  plot_marketing_expenditures <-
    ggplot(data, aes(x = 1:nrow(data), y = m)) +
    geom_line(color = "purple") +
    labs(x = "Days", y = "Amount (in Euros)") +
    ggtitle("Marketing Expenditures") +
    theme(plot.title = element_text(hjust = 0.5))
  
  return(
    list(
      plot_sales,
      plot_prices,
      plot_acquisition_costs,
      plot_marketing_expenditures
    )
  )
}

Q3.9_func(Data_Pricing)

#Question 3.10----
Q3.10_answer <- lm(s ~ p, data = Data_Pricing)
print(summary(Q3.10_answer)$coefficients)

#Question 3.13----
#Regression of price and costs:
instrument <- lm(p ~ c, data = Data_Pricing) #Instrument regression
print(summary(instrument)$coefficients)
Data_Pricing$p_hat <-
  fitted(instrument) #Calculating p_hat from instrument regression

sales_price <-
  lm(s ~ p_hat, data = Data_Pricing) #Second regression of IV, using p_hat
print(summary(sales_price)$coefficients)

#Question 3.14----
#Hausman Durbin Wu test
Hausman_Durbin_Wu_test <- function(model_IV, model_OLS, data) {
  coef_diff <- coef(model_IV) - coef(model_OLS)
  
  #Calculating the variance covariance matrix for OLS
  res_OLS <- residuals(model_OLS)
  sigma_OLS <-
    t(res_OLS) %*% res_OLS / (nrow(data))  #sigma = e'e
  matrix_OLS <- as.matrix(data[c("s", "p")])
  matrix_OLS[, 1] <- 1
  vcov_OLS <- sigma_OLS[1] * solve((t(matrix_OLS) %*% matrix_OLS))
  
  #Calculating the variance covariance  matrix for IV
  res_IV <- residuals(model_IV)
  sigma_IV <-
    t(res_IV) %*% res_IV / (nrow(data)) # sigma = e'e
  matrix_IV <- as.matrix(data[c("s", "c")])
  matrix_IV[, 1] <- 1
  Z_X_inv <- solve(t(matrix_IV) %*% matrix_OLS) # (Z'X)^(-1)
  Z_Z = t(matrix_IV) %*% matrix_IV          # (Z'Z)
  vcov_IV <-
    sigma_IV[1] * Z_X_inv %*% Z_Z %*% solve(t(matrix_OLS) %*% matrix_IV) # sigma^2 * (Z'X)^(-1)(Z'Z)(X'Z)^(-1)
  
  #Calculate the test statistic
  vc_diff <- vcov_IV - vcov_OLS
  test_statistic <-
    as.vector(t(coef_diff) %*% solve(vc_diff) %*% coef_diff)
  print(paste("Test-statistic:", test_statistic))
  print(paste(
    "p-value:",
    pchisq(
      q = test_statistic,
      df = dim(vc_diff)[1] ,
      lower.tail = F
    )
  )) # Chi square distribution with the rank as degrees of freedom
}

#Execute the HDW test
Hausman_Durbin_Wu_test(sales_price, Q3.10_answer, Data_Pricing)

#Question 3.15----
Q3.15_answer <- lm(s ~ p_hat + m, data = Data_Pricing)
print(summary(Q3.15_answer)$coefficients)

#Question 3.16----
optimal_p <-
  rep(0, nrow(Data_Pricing)) #Vector for storing optimal prices
#Retrieving coefficients of model from Question 3.15
beta <- coef(Q3.15_answer)[2]
alpha <- coef(Q3.15_answer)[1]
psi <- coef(Q3.15_answer)[3]

#Calculating optimal prices per day with derivative
optimal_p <-
  as.data.frame((Data_Pricing$c * beta - alpha - psi * Data_Pricing$m) /
                  (2 * beta))

#Expected profit for p = 4000
Expected_Profit_4000 <-
  (alpha + beta * Data_Pricing$p[759] + psi * Data_Pricing$m[759]) * (Data_Pricing$p[759] - Data_Pricing$c[759])
print(Expected_Profit_4000)

#Expected profit for p = 4001
Expected_Profit_4001 <-
  (alpha + beta * (Data_Pricing$p[759] + 1) + psi * Data_Pricing$m[759]) *
  ((Data_Pricing$p[759] + 1) - Data_Pricing$c[759])
print(Expected_Profit_4001)

#Expected extra profit when increasing price with 1 unit
Expected_Profit_4001 - Expected_Profit_4000

#Optimal p_759 = 16335.16
print(optimal_p[759,])
