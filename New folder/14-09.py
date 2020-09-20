# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:58:20 2020

@author: Nishant Agarwal
"""

#Topic ---- Dividing Data into Train and Test 
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data

from sklearn.linear_model import LinearRegression

mtcars = data('mtcars')

data=mtcars
data.head()
data.columns
data.dtypes
data.shape


model = LinearRegression()

ds= mtcars[['cyl', 'hp']]

ds


model.fit(ds['cyl'].values.reshape(-1,1), ds['hp'].values.reshape(-1,1))


plt.scatter(ds['cyl'],ds['hp'] )

r_sq = model.score(ds['cyl'].values.reshape(-1,1), ds['hp'].values.reshape(-1,1))
r_sq


ypred= model.predict(ds['cyl'].values.reshape(-1,1))
ypred


ypred1 = model.predict(np.array([7,9, 10, 12]).reshape(-1,1))

plt.scatter(ds['cyl'].values.reshape(-1,1), ds['hp'].values.reshape(-1,1) )
plt.scatter(ds['cyl'].values.reshape(-1,1), ypred)
plt.scatter(np.array([7,9, 10, 12]).reshape(-1,1), ypred1)




#%%% Sample by number

s1 = data.sample(10)
s1
#%%%
X=np.arange(10).reshape((5, 2))
y=range(5)
X
y
list(y)
X, y = np.arange(10).reshape((5, 2)), range(5)
X
list(y)
#split X and y
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= 0.33, random_state=42)
X_train
y_train
X_test
y_test
#target variable
train_test_split(y, shuffle=True)
train_test_split(y, shuffle=False)

#%%%



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd

df= pd.read_csv("D:\ML-Lab\Datasets\iris.csv")

df.columns

df.head(1)

X=df[['sepal_length', 'sepal_width', 'petal_length','petal_width']].values
X.shape



y=df[['name']]

y
y[y.name == 'virginica'] = 3
y[y.name == 'setosa'] = 1
y[y.name == 'versicolor'] = 2



y.shape
y

#these numpy objects, no head; multi-dim matrices
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=1000)

X.shape
y.shape

X_train.shape
X_test.shape
y_train.shape
y_test.shape



model = LinearRegression()

model.fit(X_train, y_train)

model.score(X_train, y_train)

y_pred= model.predict(X_test)

y_pred=np.round(y_pred,0)
y_pred
y_test



#%%%
'''
R-squared (R2) is a statistical measure that represents the proportion of the 
variance for a dependent variable that's explained by an independent variable 
or variables in a regression model. Whereas correlation explains the strength 
of the relationship between an independent and dependent variable, R-squared 
explains to what extent the variance of one variable explains the variance 
of the second variable.
'''
'''
The F value is the ratio of the mean regression sum of squares divided 
by the mean error sum of squares. Its value will range from zero to an 
arbitrarily large number. The value of Prob(F) is the probability that the 
null hypothesis for the full model is true (i.e., that all of the regression 
coefficients are zero).
'''

import statsmodels.api as sm
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
from pydataset import data





mtcars = data('mtcars')
mtcars.columns
df1 = mtcars[['wt','hp', 'mpg']]
df1.head(5)


from statsmodels.formula.api import ols

MTmodel1 = ols("mpg ~ wt + hp", data=df1).fit()


print(MTmodel1.summary())
predictionM1 = MTmodel1.predict()
predictionM1


plt.scatter()


#https://www.datarobot.com/blog/ordinary-least-squares-in-python/

'''
The CCPR plot provides a way to judge the effect of one regressor on the 
response variable by taking into account the effects of the other independent 
variables. The partial residuals plot is defined as Residuals+BiXi versus Xi.
Component-Component plus Residual (CCPR) Plots¶

The CCPR plot provides a way to judge the effect of one regressor on the 
response variable by taking into account the effects of the other independent 
variables. The partial residuals plot is defined as Residuals+BiXi  
versus Xi. The component adds BiXi versus Xi to show where the fitted line
would lie. Care should be taken if Xi is highly correlated with any of the 
other independent variables. If this is the case, the variance evident in 
the plot will be an underestimate of the true variance.

Since we are doing multivariate regressions, we cannot just look at 
individual bivariate plots to discern relationships. Instead, we want 
to look at the relationship of the dependent variable and independent 
variables conditional on the other independent variables. We can do this 
through using partial regression plots, otherwise known as added variable plots.
'''
fig, ax = plt.subplots(figsize=(12, 8))
fig = sm.graphics.plot_ccpr(MTmodel1, "wt", ax=ax)

fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(MTmodel1, fig=fig)

#%%%
#fig, ax = plt.subplots()
#fig = sm.graphics.plot_fit(MTmodel1, 0, ax=ax)
#----

from sklearn.metrics import r2_score

IV = df1[['wt','hp']].values
IV
IV.shape

DV= df1['mpg'].values
DV
DV.shape

IV_train, IV_test, DV_train, DV_test = train_test_split(IV, DV,test_size=0.2, random_state=123)

IV.shape, DV.shape
IV_train.shape, IV_test.shape, DV_train.shape, DV_test.shape


from sklearn import linear_model

MTmodel2a = linear_model.LinearRegression()


MTmodel2a.fit(IV_train, DV_train)  #putting data to model


#MTmodel2a.summary()  #no summary in sklearn
import pandas as pd
xdf=pd.DataFrame(IV_train)
xdf ['Result']=pd.DataFrame(DV_train)
xdf

from statsmodels.formula.api import ols

MTmode12a = ols('Result~1',data=xdf).fit()


MTmode12a.summary()






MTmodel2a.intercept_
MTmodel2a.coef_

predicted2a = MTmodel2a.predict(IV_test)
predicted2a

DV_test

#The mean squared error
from sklearn.metrics import mean_squared_error, r2_score

mean_squared_error(DV_test, predicted2a)
r2_score(DV_test, predicted2a)  #???

#%%%
# what to LM
# Predicting Continuous, Finding relationship between variables
# Steps : load data, split : DV & IV ; Train and test set
# Load the libraries
# create model : function + IV & DV from Train
# see r2, adjst R2, coeff, significant, other model 
# predict : Model + IV_test -> predicted_y
# rmse : predicted_y - actual_y : as less as possible
# R2 ??
# check for assumption - linear, normality, homoscedascity, multi-collinearity, auto-collinearity

#%%%% Links
#https://pythonprogramminglanguage.com/training-and-test-data/

#%%% Links
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html








#%%

#Topic: Simple Linear Regression - Area - Rent
#-----------------------------
#https://raw.githubusercontent.com/DUanalytics/pyAnalytics/master/8B-LM/lm_slr_area.py
#libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
# Load the diabetes dataset
url = "https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/slr1.csv"

url2b = "women1.csv"

data = pd.read_csv(url)

#data = pd.read_csv('data/slr1.csv')

data

#data has features, target has DV value Use only one feature

X = data.X.values
X
X=X.reshape(-1,1)
X
y = data.Y.values
y=y.reshape(-1,1)
y

#%%%

from sklearn import linear_model


lm = linear_model.LinearRegression()
model1 = lm.fit(X, y)
print(model1)
model1.score(X,y)  #R2





#Coefficients
model1.coef_   #b1 coef
model1.intercept_ #b0 coef
y_pred1 = model1.predict(X)
y_pred1






#The mean squared error
mean_squared_error(y,y_pred1)

r2_score(y, y_pred1)

print('Variance score: %.2f' % r2_score(y, y_pred1))
# Plot outputs  : select all at once and run
plt.scatter(X, y,  color='black')
plt.plot(X, y_pred1, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show();


#%% Model 2
import statsmodels.api as sm
X,y
model2 = sm.OLS(X, y).fit()
model2.summary()

predictions2 = model2.predict(X)
predictions2
model2.summary()
# add constant
X2 = sm.add_constant(X) #Adds a column of ones to an array
model3 = sm.OLS(y, X2).fit() #output, input
model3.summary()
predictions3 = model3.predict(X2)
predictions3

#%% Model4
#https://www.learndatasci.com/tutorials/predicting-housing-prices-linear-regression-using-python-pandas-statsmodels/

from statsmodels.formula.api import ols
data.columns
model4 = ols('Y ~ X', data=data).fit()
model4.summary()
#diagnostic plots
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model4,"X", ax=ax)
ax.set_ylabel("Rent")
ax.set_xlabel("Area")
ax.set_title("Linear Regression")
plt.show();








#%%

#Topic: Linear Regression Stock Market Prediction 
#-----------------------------
#libraries
import pandas as pd
import matplotlib.pyplot as plt

Stock_Market = {'Year': [2017,2017,2017, 2017,2017,2017,2017,2017, 2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016, 2016,2016,2016], 'Month': [12, 11,10,9,8,7,6, 5,4,3, 2,1,12,11, 10,9,8,7,6,5,4,3,2,1], 'Interest_Rate': [2.75,2.5,2.5,2.5,2.5, 2.5,2.5,2.25,2.25, 2.25,2,2,2,1.75,1.75, 1.75,1.75, 1.75,1.75,1.75,1.75,1.75,1.75,1.75], 'Unemployment_Rate': [5.3,5.3, 5.3,5.3,5.4,5.6,5.5, 5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1, 5.9,6.2,6.2, 6.1],'Stock_Index_Price':[1464,1394,1357,1293,1256,1254,1234,1195, 1159,1167,1130,1075,1047,965, 943,958,971,949,884,866,876,822,704,719]   }  #dictionary format
type(Stock_Market)

df = pd.DataFrame(Stock_Market, columns=['Year','Month','Interest_Rate', 'Unemployment_Rate','Stock_Index_Price']) 
df.head()
print (df)

#check that a linear relationship exists between the:
#Stock_Index_Price (dependent variable) and Interest_Rate (independent variable)
#Stock_Index_Price (dependent variable) and Unemployment_Rate (independent variable)

#run these lines together
plt.scatter(df['Interest_Rate'], df['Stock_Index_Price'], color='red')
plt.title('Stock Index Price Vs Interest Rate', fontsize=14)
plt.xlabel('Interest Rate', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show();
#

 linear relationship exists between the Stock_Index_Price and the Interest_Rate. Specifically, when interest rates go up, the stock index price also goes up:
    
    
    
plt.scatter(df['Unemployment_Rate'], df['Stock_Index_Price'], color='green')
plt.title('Stock Index Price Vs Unemployment Rate', fontsize=14)
plt.xlabel('Unemployment Rate', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show() ;   

#linear relationship also exists between the Stock_Index_Price and the Unemployment_Rate – when the unemployment rates go up, the stock index price goes down (here we still have a linear relationship, but with a negative slope):

#Multiple Linear Regression
from sklearn import linear_model #1st method
import statsmodels.api as sm  #2nd method
    
X = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example. Alternatively, you may add additional variables within the brackets
Y = df['Stock_Index_Price']
 

data1=X
data1['R']=Y
data1





# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)


y_pred= regr.predict(X.values)

import statsmodels.api as sm  #2
X,y

model2= ols("R~Interest_Rate+Unemployment_Rate" ,data=data1).fit()
model2.summary()




from sklearn.metrics import mean_squared_error, r2_score

r2_score(y, y_pred)



New_Interest_Rate = 2.75
New_Unemployment_Rate = 5.3
print ('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))




















