# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:36:29 2020

@author: Nishant Agarwal
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

url=pd.read_csv("C:\\Users\\Nishant Agarwal\\Desktop\\New folder\\book1.csv")


url.head()
url.columns
url.time
url['time']


X = url['time'].values.reshape(-1,1)
y = url['page']
y
X.shape, y.shape
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2,random_state=1000)
X_train.shape, X_test.shape
y_train.shape, y_test.shape



logistic_regression= LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred=logistic_regression.predict(X_test)

y_pred
y_test

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap='YlGnBu',fmt='g')

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
