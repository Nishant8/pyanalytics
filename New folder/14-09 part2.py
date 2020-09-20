# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:09:53 2020

@author: Nishant Agarwal
"""

#Topic:Logistic Regression
#-----------------------------
#libraries

#In general, a binary logistic regression describes the relationship between the dependent binary variable and one or more independent variable/s.

#The binary dependent variable has two possible outcomes:
#‘1’ for true/success; or
#‘0’ for false/failure

#case : let’s say that your goal is to build a logistic regression model in Python in order to determine whether candidates would get admitted to a prestigious university.
#there are two possible outcomes: Admitted (represented by the value of ‘1’) vs. Rejected (represented by the value of ‘0’).

#libraries
import pandas as pd # used to create the DataFrame to capture the dataset in Python
#sklearn    # used to build the logistic regression model in Python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns # used to create the Confusion Matrix
import matplotlib.pyplot as plt # used to display charts

#data from csv
# url = "https://stats.idre.ucla.edu/stat/data/binary.csv"
# df = pd.read_csv(url)
# df

#data
gmat =  [780,750,690,710,680,730,690,720,740, 690,610,690,710,680, 770,610,580, 650,540, 590,620, 600,550,550, 570,670,660,580,650,660,640,620,660, 660,680,650,670,580,590,690]
gpa =  [4,3.9,3.3,3.7,3.9,3.7, 2.3,3.3,3.3,1.7,2.7, 3.7,3.7,3.3,3.3, 3,2.7,3.7,2.7,2.3, 3.3, 2,2.3,2.7,3,3.3,3.7,2.3, 3.7,3.3,3,2.7, 4,3.3,3.3,2.3,2.7,3.3,1.7, 3.7]
work_experience = [3,4,3,5,4,6,1,4,5,1,3,5, 6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6, 5,1,2,4,6,5,1,2,1,4,5]
admitted = [1,1,0,1, 0,1,0,1,1,0, 0,1,1,0,1,0,0,1, 0,0,1,0,0,0,0,1,1,0, 1,1,0,0,1,1,1, 0,0,0,0,1]

candidates = {'gmat':gmat, 'gpa': gpa, 'work_experience': work_experience ,'admitted': admitted  }
candidates
type(candidates)

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
df

df.head()
df.dtypes

#set the independent variables (represented as X) and the dependent variable (represented as y):

X = df[['gmat', 'gpa','work_experience']] #array
y = df['admitted']
X, y

#split data : Then, apply train_test_split. For example, you can set the test size to 0.25, and therefore the model testing will be based on 25% of the dataset, while the model training will be based on 75% of the dataset:

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
X_train.shape, X_test.shape
y_train.shape, y_test.shape





#apply logistic regression
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)

y_pred=logistic_regression.predict(X_test)


y_pred

y_test
#print the Accuracy and plot the Confusion Matrix:

    
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))


y_test
y_pred


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap='coolwarm', annot_kws={'size':20}, cbar=False)
plt.show();

# TP = True Positives = 4
# TN = True Negatives = 4
# FP = False Positives = 1
# FN = False Negatives = 1


'''

TP
TN
FP
FN

(TP+TN)/(TP+TN+FP+FN)

'''


Accuracy = (4+4)/10  #(TP+TN)/Total .8
Accuracy

#Confusion Matrix with an Accuracy of 0.8 (may vary with sklearn version)

print (X_test) #test dataset
#original dataset (from step 1) had 40 observations. Since we set the test size to 0.25, then the confusion matrix displayed the results for 10 records (=40*0.25). These are the 10 test records:
print (y_pred) #predicted values
#The prediction was also made for those 10 records (where 1 = admitted, while 0 = rejected):
y_test, y_pred
type(y_test), type(y_pred)

    
#%%predict on new data set
#use the existing logistic regression model to predict whether the new candidates will get admitted. The new set of data can then be captured in a second DataFrame called df2:
    
new_candidates = {'gmat': [590,740,680,610,710], 'gpa': [2,3.7,3.3,2.3,3], 'work_experience': [3,4,6,1,5] }
new_candidates
df2 = pd.DataFrame(new_candidates,columns= ['gmat', 'gpa','work_experience'])
df2

y_pred2=logistic_regression.predict(df2)
print (df2)
print (y_pred2)


#The first and fourth candidates are not expected to be admitted, while the other candidates are expected to be admitted.
#df.concat(y_pred2)
pd.concat([df2, pd.Series(y_pred2)], axis=1, sort=False)
df2











#%%
#Topic: Logistic Regression - Python - Diabetes
#-----------------------------
#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python
#data
url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/pima-indians-diabetes.csv'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv(url, header=None, names=col_names)

pima.head()

#%%%Selecting Feature
#Here, you need to divide the given columns into two types of variables dependent(or target variable) and independent variable(or feature variables).
#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']

pima.columns

X = pima[feature_cols] # Features
X
y = pima.label # Target variable
y

#%%%Splitting Data
#To understand model performance, dividing the dataset into a training set and a test set is a good strategy.
#Let's split dataset by using function train_test_split(). You need to pass 3 parameters features, target, and test_set size. Additionally, you can use random_state to select records randomly.
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.1, random_state=0)
#Here, the Dataset is broken into two parts in a ratio of 75:25. It means 75% data will be used for model training and 25% for model testing.

#%%%
#Model Development and Prediction
#First, import the Logistic Regression module and create a Logistic Regression classifier object using LogisticRegression() function.
#Then, fit your model on the train set using fit() and perform prediction on the test set using predict().
# import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
y_pred


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


metrics.accuracy_score(y_test, y_pred)




chk= [[1,68,63.1,33,137,40,2.288]]
y_val= logreg.predict(chk)
y_val

#%%%Model Evaluation using Confusion Matrix
#A confusion matrix is a table that is used to evaluate the performance of a classification model. You can also visualize the performance of an algorithm. The fundamental of a confusion matrix is the number of correct and incorrect predictions are summed up class-wise.
# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
(119 + 36)/(119 + 36 + 26 + 11)
#Here, you can see the confusion matrix in the form of the array object. The dimension of this matrix is 2*2 because this model is binary classification. You have two classes 0 and 1. Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions. In the output, 119 and 36 are actual predictions, and 26 and 11 are incorrect predictions.


#%%Visualizing Confusion Matrix using Heatmap
#Let's visualize the results of the model in the form of a confusion matrix using matplotlib and seaborn.
#Here, you will visualize the confusion matrix using Heatmap.
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline # for Jupiter
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show();


#%%%Confusion Matrix Evaluation Metrics
#Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall.

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


#classification rate of 80%, considered as good accuracy.
#Precision: Precision is about being precise, i.e., how accurate your model is. In other words, you can say, when a model makes a prediction, how often it is correct. In your prediction case, when your Logistic Regression model predicted patients are going to suffer from diabetes, that patients have 76% of the time.
#Recall: If there are patients who have diabetes in the test set and your Logistic Regression model can identify it 58% of the time.


#%%ROC Curve
#Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. 
#It shows the tradeoff between sensitivity and specificity.
#A receiver operating characteristic curve, or ROC curve, is a graphical plot that
# illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
#The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) 
#at various threshold settings.
# The true-positive rate is also known as sensitivity, recall or probability of detection[7] in machine learning. 
#The false-positive rate is also known as probability of false alarm[7] 


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();
auc
#AUC score for the case is 0.86. AUC score 1 represents perfect classifier, and 0.5 represents a worthless classifier.




#%%%Advantages
#Because of its efficient and straightforward nature, doesn't require high computation power, easy to implement, easily interpretable, used widely by data analyst and scientist. Also, it doesn't require scaling of features. Logistic regression provides a probability score for observations.
#%%#Disadvantages
#Logistic regression is not able to handle a large number of categorical features/variables. It is vulnerable to overfitting. Also, can't solve the non-linear problem with the logistic regression that is why it requires a transformation of non-linear features. Logistic regression will not perform well with independent variables that are not correlated to the target variable and are very similar or correlated to each other.
#%% Other Points
#You don't need to scale data for logistic regression because logistic regression coefficients represent the effect of one unit change in the independent variable on the dependent variable(log odd). If we scale the data between range 0-1 than unit change will shift the value from low to high but there is no change in log odd values.If you are using logistic regression with regularization than it is recommended normalize.  














#%%
#SIR EXCERICES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/pima-indians-diabetes.csv'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']


pima = pd.read_csv(url, header=None, names=col_names)
pima.head()
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
pima.columns

X = pima[feature_cols]
X
y = pima.label
y 

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.1)
X_train.shape, X_test.shape
y_train.shape, y_test.shape

logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

y_pred
y_test

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap='YlGnBu',fmt='g')

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))


#%%







#Topic:Logistics Regression  - Cut off Values
#-----------------------------
#libraries
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.metrics import roc_curve, auc

# read the data in
url = "https://stats.idre.ucla.edu/stat/data/binary.csv"
df = pd.read_csv(url)
#df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

# rename the 'rank' column because there is also a DataFrame method called 'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]
# dummify rank
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
# create a clean data frame for the regression
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep]

# manually add the intercept
data['intercept'] = 1.0

train_cols = data.columns[1:]
# fit the model
result = sm.Logit(data['admit'], data[train_cols]).fit()
result.summary()

# Add prediction to dataframe
data['pred'] = result.predict(data[train_cols])

fpr, tpr, thresholds =roc_curve(data['admit'], data['pred'])
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'])
pl.plot(roc['1-fpr'], color = 'red')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])






