# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:23:21 2020

@author: Nishant Agarwal
"""

#Topic:Linear Model - Steps
#-----------------------------
#libraries

#S1- libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#S2 - data
x = np.array([5,15,25,35,45,55]).reshape((-1,1))  #making 2 dim
x  #IV
y = np.array([5,20,14,32,22, 38])  #1 dim
y #DV #(y can 2 dim also : y.reshape((-1,1)))


plt.scatter(x,y)


#THEORY
#Y = Intercept + (Slope*X)  => Linear Regression Line




#S3 - Model
model = LinearRegression()  #create blank model




#other options- fit_intercept (y/N), normalise i/p var
model.fit(x,y)  #find optimal values of weights b0, b1 using x & y, .fit() fits the model

model = LinearRegression().fit(x,y) #another way  #2 lines into 1
model


#S4 - Results
r_sq = model.score(x, y)

r_sq #coeff of determination : > .6 is good



model.intercept_  #bo

model.coef_  #b1 #Slope


y = 5.6 + .54 * x  #mathematical equation
y
#if x is increased by 1 units, y increased by .54 units; when x=0, y=5.6 (constant value)


#S5 Predict
y_pred = model.predict(x)  #predict on trained data 
y_pred
print(y_pred, sep='\t ')



plt.scatter(x,y)
plt.scatter(x, y_pred)


x_pred=np.array([65,70,75]).reshape((-1,1))
y_pred1=model.predict(x_pred)
plt.scatter(x_pred, y_pred1)



plt.scatter(x,y)
plt.scatter(x, y_pred)
plt.scatter(x_pred, y_pred1)



y_pred2 = model.intercept_ + model.coef_ * x
print(y_pred2, sep='\t ')


#new values
x_new = np.arange(5).reshape((-1,1))
x_new
y_new = model.predict(x_new)
print(y_new, sep ='\t ')


#%% MUltiple Linear Regression
import numpy as np
from sklearn.linear_model import LinearRegression
x = [[0,1], [5,1], [15,2], [25,2], [35,11], [45,15], [55,34], [60,35]]
x
y = [4,5,20,14,32,22,38,43]
y
x, y = np.array(x), np.array(y)
x #2 dim ; MLR - 2 variable, 2 dim(LxB) with 2 columns
y #1 dim
x.shape, y.shape

#S3: Model & Fit
model = LinearRegression().fit(x,y)
model.score(x,y)  #R2 
model.intercept_ # constant
model.coef_ #b0, b1
#keeping one IV constant(x1), if x2 increases by 1 unit, y increases by .28 units and so on

#S4 : predict
y_pred = model.predict(x)
y_pred
y_pred2 = model.intercept_ + np.sum(model.coef_ * x, axis=1)
y_pred2
y_pred - y_pred2

#new data
x_new = np.arange(10). reshape((-1,2))
x_new
y_new = model.predict(x_new)
y_new



#%% Stats Models

import numpy as np
import statsmodels.api as sm

from statsmodels.tools import add_constant
x = [[0,1], [5,1], [15,2], [25,2], [35,11], [45,15], [55,34], [60,35]]
x
y = [4,5,20,14,32,22,38,43]
y

x= sm.add_constant(x)  #constant term of 1 added
x
model3 = sm.OLS(y,x)
model3
results = model3.fit()
results
results.summary()
results.rsquared  #coeff of determination
results.rsquared_adj 
results.params  #bo, b1, b2

results.fittedvalues
results.predict(x)


#%%AIC & BIC  
#https://pypi.org/project/RegscorePy
#pip install RegscorePy
import RegscorePy
#aic(y, y_pred, p)
RegscorePy.aic.aic(y=y, y_pred= results.predict(x), p=1)
RegscorePy.bic.bic(y=y, y_pred= results.predict(x), p=1)







#%%

#Topic: Linear Regression 2 Media Coy Case
#-----------------------------
#libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#read data
#media = pd.DataFrame(pd.read_csv('data/mediacompany.csv'))
url ='https://raw.githubusercontent.com/DUanalytics/pyAnalytics/master/data/mediacompany.csv'

media = pd.DataFrame(pd.read_csv(url))

media.head()

#check duplicates
sum(media.duplicated(subset = 'Date')) == 0
#if duplicates value will not be =0

#remove last column
media = media.drop('Unnamed: 7', axis=1)
media.head()

#description
media.describe
media.shape
media.info()

#null values check
media.isnull().sum()  #each col

#outlier analysis
media.dtypes
fig, axs = plt.subplots(2,2, figsize = (10,5))
plt1 = sns.boxplot(media['Views_show'], ax = axs[0,0])
plt1 = sns.boxplot(media['Visitors'], ax = axs[0,1])
plt1 = sns.boxplot(media['Views_platform'], ax = axs[1,0])
plt1 = sns.boxplot(media['Ad_impression'], ax = axs[1,1])
plt.tight_layout()
plt.show();

#date format
media['Date'] = pd.to_datetime(media['Date'], dayfirst = False)
media.dtypes
media.head

#weekday from date
media['Day_of_week'] = media['Date'].dt.dayofweek
media.head()

#exploratory Data Analysis

#Views Show
sns.boxplot(media['Views_show'])

#univariate analysis
media.plot.line(x='Date', y='Views_show')

#DOW
sns.barplot(data = media, x='Day_of_week', y='Views_show')
#which week it was highest - Sun, Sat
#weekdays and weekend(1)
di={5:1, 6:1, 0:0, 1:0, 2:0, 3:0, 4:0} #dictionary
media['weekend'] = media['Day_of_week'].map(di)
media.head()

#weekends
sns.barplot(data = media, x='weekend', y='Views_show')
#higher on weekends

#Ad impressions
ax = media.plot(x='Date', y='Views_show', legend=False)
ax2 = ax.twinx()
media.plot(x='Date', y='Ad_impression', ax=ax2, legend=False, color='r')
ax.figure.legend()
plt.show()

sns.scatterplot(data=media, x='Ad_impression', y='Views_show')

#visitors
sns.scatterplot(data=media, x='Visitors', y='Views_show')

#Views platform
sns.scatterplot(data=media, x='Views_platform', y='Views_show')
#some views are some what proportional related to Platform views

#cricket match
sns.barplot(data = media, x='Cricket_match_india', y='Views_show')
#drop in views in some shows due to cricket match

#character A
sns.barplot(data = media, x='Character_A', y='Views_show')
#presence of Character A improves show viewership


#%%% Model Building
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#scale data ex Yes/No, Ad_impression
num_vars = ['Views_show', 'Visitors', 'Views_platform', 'Ad_impression']
media[num_vars] = scaler.fit_transform(media[num_vars])
media.head()

sns.heatmap(media.corr(), annot=True)

#Model1
X= media[['Visitors', 'weekend']]
X
y = media['Views_show']

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,y)
lm.score(X,y)
lm.coef_

#different library: Constant term has to be added here
import statsmodels.api as sm
#need to fit constant
X = sm.add_constant(X)
lm_1 = sm.OLS(y, X).fit()
print(lm_1.summary())
#significnt variables - weekends, Visitors. P>|t| : .05 

#2nd model : keep changing the combination of variables 
X= media[{'Visitors', 'weekend', 'Character_A'}]
y=media['Views_show']
import statsmodels.api as sm
X=sm.add_constant(X)
lm_2 = sm.OLS(y, X).fit()
print(lm_2.summary())



#like this keep creating model. Model which has higher AdjR2 is better.
#see more from here
#(https://www.kaggle.com/ashydv/media-company-case-study-linear-regression)
















#%%
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



model=LinearRegression()



ds=mtcars[['hp', 'drat']]

ds


model.fit(ds['hp'].values.reshape(-1,1),ds['drat'].values.reshape(-1,1))

plt.scatter(ds['hp'],ds['drat'])


r_sq = model.score(ds['hp'].values.reshape(-1,1),ds['drat'].values.reshape(-1,1))

r_sq



#Example

ds=mtcars[['cyl', 'hp']]

ds


model.fit(ds['cyl'].values.reshape(-1,1),ds['hp'].values.reshape(-1,1))

plt.scatter(ds['cyl'],ds['hp'])


r_sq = model.score(ds['cyl'].values.reshape(-1,1),ds['hp'].values.reshape(-1,1))

r_sq





ypred= model.predict(ds['cyl'].values.reshape(-1,1))
ypred


y_pred1=model.predict(np.array([7,9,10,12]).reshape(-1,1))




plt.scatter(ds['cyl'].values.reshape(-1,1),ds['hp'].values.reshape(-1,1))
plt.scatter(ds['cyl'].values.reshape(-1,1), ypred)
plt.scatter(np.array([7,9,10,12]).reshape(-1,1), y_pred1)











# Same example as above but change position of X & Y

ds=mtcars[['hp', 'cyl']]

ds


model.fit(ds['hp'].values.reshape(-1,1),ds['cyl'].values.reshape(-1,1))

plt.scatter(ds['hp'],ds['cyl'])


r_sq = model.score(ds['hp'].values.reshape(-1,1),ds['cyl'].values.reshape(-1,1))

r_sq







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
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset.

iris = load_iris()
type(iris)  #Bunch

iris

iris.columns

X=iris.data
y=iris.target




X.shape



y.shape

#these numpy objects, no head; multi-dim matrices


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=1000)

X_train.shape
X_test.shape
y_train.shape
y_test.shape



model=LinearRegression()

model.fit(X_train, y_train)

model.score(X_train, y_train)


y_pred= model.predict(X_test)
y_pred

y_pred=np.round(y_pred, 0)

y_pred
y_test


#%% split data into training and test data.- specify train and test size


from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

regr=linear_model.LinearRegression()

regr.fit(X_train, y_train)


test_pred=regr.predict(X_test)     

test_pred.shape

test_pred


print("Mean squared error: %.2f"
      % mean_squared_error( y_test, test_pred ))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, test_pred ))


'''

# Plot outputs  : select all at once and run
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, test_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
'''



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
from statsmodels.formula.api import ols

from pydataset import data
mtcars = data('mtcars')
mtcars.columns
df1 = mtcars[['wt','hp', 'mpg']]
df1.head(5)

MTmodel1 = ols("mpg ~ wt + hp", data=df1).fit()
print(MTmodel1.summary())
predictionM1 = MTmodel1.predict()
predictionM1

#%%%
'''
Residual = Observed – Predicted
'''


fig= plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(MTmodel1,"wt", fig=fig)


#%%%
#This creates one graph with the scatterplot of observed values compared to 
#fitted values.

fig= plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(MTmodel1,"hp", fig=fig) 
#%%%%
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

DV= df1['mpg'].values
DV


IV_train, IV_test, DV_train, DV_test = train_test_split(IV, DV,test_size=0.2, random_state=123)

IV.shape, DV.shape
IV_train.shape, IV_test.shape, DV_train.shape, DV_test.shape

from sklearn import linear_model
MTmodel2a = linear_model.LinearRegression()
MTmodel2a.fit(IV_train, DV_train)  #putting data to model
#MTmodel2a.summary()  #no summary in sklearn
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
















































































