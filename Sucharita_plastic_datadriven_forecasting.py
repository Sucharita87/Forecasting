# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:17:37 2020

@author: SUCHARITA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
from datetime import datetime,time
#from sm.tsa.statespace import sa
plastic = pd.read_csv("F:\\ExcelR\\Assignment\\Forecasting\\PlasticSales.csv")
plastic.Sales.plot()
plastic["Date"] = pd.to_datetime(plastic.Month,format="%b-%y")

plastic["month"] = plastic.Date.dt.strftime("%b") # month extraction

plastic["year"] = plastic.Date.dt.strftime("%Y") # year extract
heatmap_y_month = pd.pivot_table(data=plastic,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot 
sns.boxplot(x="month",y="Sales",data=plastic) # 2 outliers in nov and dec, sales inc during second half of the year
sns.boxplot(x="year",y="Sales",data=plastic) # sales increased over the years

# Line plot 
sns.lineplot(x="year",y="Sales",data=plastic) # increasing trend


# Centering moving average for the time series to understand better about the trend character in plastic sales
plastic.Sales.plot(label="org")
for i in range(2,24,6):
    plastic["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(plastic.Sales,model="additive",freq=12)
decompose_ts_add.plot() # trend is positive/increasing, seasonality is swinging between positive and negative values
# residuals have a pattern and are swinging between positive and negative ranges are increasing by the end of thedata set i.e with progressing years
decompose_ts_mul = seasonal_decompose(plastic.Sales,model="multiplicative",freq=12)
decompose_ts_mul.plot()# trend is positive/increasing, seasonality is in positive range
# residuals  are almost constant

# ACF plots and PACF plots on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(plastic.Sales,lags=12)
tsa_plots.plot_pacf(plastic.Sales,lags=12)

# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = plastic.head(50)
Test = plastic.tail(10)
# change the index value in pandas data frame 
Test.set_index(np.arange(1,11),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) # 29.7

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) # 25.9



# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales) # 25.69

hwe_model_add_add = ExponentialSmoothing(plastic["Sales"],seasonal="add",trend="add",seasonal_periods=12).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = plastic.index[0],end = plastic.index[-1])
pred_hwe_add_add

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales) # 25.29


# Lets us use auto_arima from p
#not working : from pyramid.arima import auto_arima

from pmdarima.arima import auto_arima

auto_arima_model = auto_arima(Train["Sales"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=12,start_P=0,seasonal=True, 
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)
                
            
auto_arima_model.summary() 
# AIC ==> 391.6
# BIC ==> 398

# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )

# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=10))
# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
MAPE(pred_test,Test.Sales)   # 13.63, least MAPE value is the best
