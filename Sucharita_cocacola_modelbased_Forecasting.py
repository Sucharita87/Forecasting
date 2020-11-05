# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 00:07:55 2020

@author: SUCHARITA
"""

import pandas as pd
c= pd.read_excel("F:\\ExcelR\\Assignment\\Forecasting\\CocaCola_Sales_Rawdata.xlsx")
quarter =['q1','q2','q3','q4'] 
import numpy as np
coke=c[0:40] # as q3 and q4 is missing from the last year so we remove q1 and q2 as well for the same year
coke[0:40]
q = coke["Quarter"][0]
q[0:2] # 1st 2 letters of the column "Months" will be printed
coke['quarter']= 0

for i in range(40):
    q = coke["Quarter"][i] 
    coke['quarter'][i]= q[0:2]
    
month_dummies = pd.DataFrame(pd.get_dummies(coke['quarter']))
coke1 = pd.concat([coke,month_dummies],axis = 1)
coke1.quarter.value_counts()
coke1["t"] = np.arange(1,41)
coke1["t_squared"] = coke1["t"]*coke1["t"]
coke1.columns
coke1.Sales.plot() #high increase with time, upward trend
coke1["log_sales"] = np.log(coke1["Sales"])
Train = coke1.head(32)
Train.quarter.value_counts()
Test = coke1.tail(8)
Test.quarter.value_counts()
Test.set_index(np.arange(1,9))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear # 714.33

##################### Exponential ##############################

Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp # 577.97

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad # 413.34

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea # 1735.62

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad  # 228.75

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea # 1793.29

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea  # 437.78

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
#               MODEL  RMSE_Values
#0        rmse_linear   714.339771
#1           rmse_Exp   577.974532
#2          rmse_Quad   413.340658
#3       rmse_add_sea  1735.625319
#4  rmse_add_sea_quad   228.750502
#5      rmse_Mult_sea  1793.293090
#6  rmse_Mult_add_sea   437.788155

# so rmse_Quad has the least value among the models prepared so far 
# we created 4 dummy variables in form of 4 quarters in a year

model_full = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_new = pd.Series(Quad.predict(Test[["t","t_squared"]]))
pred_new

