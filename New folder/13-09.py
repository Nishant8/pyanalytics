# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:05:27 2020

@author: Nishant Agarwal
"""

import matplotlib.pyplot as plt
import numpy as np


mu, sigma=65,10

s=np.random.normal(mu, sigma, 100)
s




plt.figure(1,figsize=(5,5),dpi=300)
count,bins, ignored= plt.hist(s,10)
bins
count
ignored






# -*- coding: utf-8 -*-
#binomial distribution
#The binomial distribution model deals with finding the probability of success of an event which has only two possible outcomes in a series of experiments. For example, tossing of a coin always gives a head or a tail. The probability of finding exactly 3 heads in tossing a coin repeatedly for 10 times is estimated during the binomial distribution.
#We use the seaborn python library which has in-built functions to create such probability distribution graphs. Also, the scipy package helps is creating the binomial distribution.
#Eg1
from scipy.stats import binom
import seaborn as sb


data_binom=binom.rvs(size=10,n=20,p=0.8)


ax = sb.distplot(data_binom,  kde=True,  hist_kws= {"linewidth": 25,'alpha':1})
ax.set(xlabel='Binomial', ylabel='Frequency')






data_binom=binom.rvs(size=10,n=20,p=0.1)


ax = sb.distplot(data_binom,  kde=True,  hist_kws= {"linewidth": 25,'alpha':1})
ax.set(xlabel='Binomial', ylabel='Frequency')





data_binom=binom.rvs(size=10,n=20,p=1)


ax = sb.distplot(data_binom,  kde=True,  hist_kws= {"linewidth": 25,'alpha':1})
ax.set(xlabel='Binomial', ylabel='Frequency')





'''
rvs(n, p, loc=0, size=1)	Random variates.
pmf(x, n, p, loc=0)	Probability mass function.
logpmf(x, n, p, loc=0)	Log of the probability mass function.
cdf(x, n, p, loc=0)	Cumulative density function.
logcdf(x, n, p, loc=0)	Log of the cumulative density function.
median(n, p, loc=0)	Median of the distribution.
mean(n, p, loc=0)	Mean of the distribution.
var(n, p, loc=0)	Variance of the distribution.
std(n, p, loc=0)	Standard deviation of the distribution.

Parameters:	
x : array_like quantiles
q : array_like lower or upper tail probability
n, p : array_like shape parameters
loc : array_like, optional
location parameter (default=0)
size : int or tuple of ints, optional
shape of random variates (default computed from input arguments )
'''

data_binom = binom.rvs(n=1,p=0.1,loc=1,size=1000)
data_binom




import collections
import numpy as np





collections.Counter(data_binom)





data_binom = binom.rvs(n=20,p=0.8,loc=1,size=1000)
ax = sb.distplot(data_binom,  kde=True,  hist_kws= {"linewidth": 25,'alpha':0.7})
ax.set(xlabel='Binomial', ylabel='Frequency')



data_binom = binom.rvs(n=20,p=0.5,size=10)
data_binom
ax = sb.distplot(data_binom,  kde=True,  hist_kws= {"linewidth": 25,'alpha':0.7})
ax.set(xlabel='Binomial', ylabel='Frequency')



data_binom = binom.rvs(n=20,p=0.1,size=10)
data_binom
ax = sb.distplot(data_binom,  kde=True,  hist_kws= {"linewidth": 25,'alpha':0.7})
ax.set(xlabel='Binomial', ylabel='Frequency')




data_binom = binom.rvs(n=20,p=0.9,size=10)
data_binom
ax = sb.distplot(data_binom,  kde=True,  hist_kws= {"linewidth": 25,'alpha':0.7})
ax.set(xlabel='Binomial', ylabel='Frequency')




#import libraries

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt





#assuming value
total_sample=20
p_value=0.5
n=np.arange(0,20)
n

#import binomial function
#pmf -probability mass functions
binomial=stats.binom.pmf(n,total_sample,p_value)
binomial


binomial.sum()

#plot the binomial distribution

plt.plot(n,binomial)
plt.title('Binomial distribution')
plt.xlabel('number of success')
plt.ylabel('probability of success')
plt.show()






#sir explanation



data_binom = binom.rvs(n=20,p=0.5,size=20)
data_binom

plt.plot(n,data_binom)
plt.title('Binomial distribution')
plt.xlabel('number of success')
plt.ylabel('probability of success')
plt.show()




data_binom = binom.rvs(n=20,p=0.5,size=10)
data_binom

plt.plot(n,data_binom)
plt.title('Binomial distribution')
plt.xlabel('number of success')
plt.ylabel('probability of success')
plt.show()






# -*- coding: utf-8 -*-

#Poision Distribution
#A Poisson distribution is a distribution which shows the 
#likely number of times that an event will occur within a 
#pre-determined period of time. It is used for independent 
#events which occur at a constant rate within a given interval 
#of time. The Poisson distribution is a discrete function, meaning
# that the event can only be measured as occurring or not as
# occurring, meaning the variable can only be measured in whole numbers.
##We use the seaborn python library which has in-built functions 
#to create such probability distribution graphs. Also the scipy 
#package helps is creating the binomial distribution.

from scipy.stats import poisson
import seaborn as sb
import matplotlib.pyplot as plt




plt.figure(1,figsize=(5,5),dpi=72)


data_poisson = poisson.rvs(mu=4, size=10000)
data_poisson


import pandas as pd

pd.Series(data_poisson)






ax = sb.distplot(data_binom,   kde=True, color='green', hist_kws= {"linewidth": 25,'alpha':1})
ax.set(xlabel='Poisson', ylabel='Frequency')





#Topic: Statistics - Covariance
#-----------------------------
#Covariance provides the a measure of strength of correlation between two variable or more set of variables. The covariance matrix element Cij is the covariance of xi and xj. The element Cii is the variance of xi.
#If COV(xi, xj) = 0 then variables are uncorrelated
#If COV(xi, xj) > 0 then variables positively correlated
#If COV(xi, xj) > < 0 then variables negatively correlated

#numpy.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)

#Parameters:
#m : [array_like] A 1D or 2D variables. variables are columns
#y : [array_like] It has the same form as that of m.
#rowvar : [bool, optional] If rowvar is True (default), then each row represents a variable, with observations in the columns. Otherwise, the relationship is transposed:
#bias : Default normalization is False. If bias is True it normalize the data points.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



A = [45,37,42,35,39]
B = [38,31,26,28,33]
C = [10,15,17,21,12]


data = np.array([A,B,C])

data


covMatrix = np.cov(data,bias=True)  # No bias is given then it will take false value as default
print (covMatrix)


sns.heatmap(covMatrix, annot=True, fmt='g')
plt.show()





#Topic:Correlation, Covariance,
#-----------------------------
#libraries

#The difference between variance, covariance, and correlation is:

#Variance is a measure of variability from the mean
#Covariance is a measure of relationship between the variability of 2 variables - covariance is scale dependent because it is not standardized
#Correlation is a of relationship between the variability of of 2 variables - correlation is standardized making it not scale dependent


import pandas as pd
import numpy as np


df = pd.DataFrame(np.random.randint(low= 0, high= 20, size= (5, 2)),  columns= ['Commercials Watched', 'Product Purchases'])
df
df.agg(["mean", "std"])
df.cov()
df.corr()


#skewness & Kurtosis
#%matplotlib inline
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew

import matplotlib.pyplot as plt

plt.style.use('ggplot')

data = np.random.normal(0, 1, 10000000)
np.var(data)

plt.hist(data, bins=60)

print("mean : ", np.mean(data))
print("var  : ", np.var(data))
print("skew : ",skew(data))
print("kurt : ",kurtosis(data))



#%%

import numpy as np
from scipy.stats import kurtosis, skew

x_random = np.random.normal(0, 2, 10000)

plt.hist(x_random)

print("mean : ", np.mean(x_random))
print("var  : ", np.var(x_random))
print("skew : ",skew(x_random))
print("kurt : ",kurtosis(x_random))



x_binom=np.random.binomial(size=10000,n=2,p=.8)

plt.hist(x_binom)

print("mean : ", np.mean(x_binom))
print("var  : ", np.var(x_binom))
print("skew : ",skew(x_binom))
print("kurt : ",kurtosis(x_binom))


x_binom=np.random.binomial(size=10000,n=2,p=1)

plt.hist(x_binom)

print("mean : ", np.mean(x_binom))
print("var  : ", np.var(x_binom))
print("skew : ",skew(x_binom))
print("kurt : ",kurtosis(x_binom))


x_binom=np.random.binomial(size=10000,n=2,p=.5)

plt.hist(x_binom)

print("mean : ", np.mean(x_binom))
print("var  : ", np.var(x_binom))
print("skew : ",skew(x_binom))
print("kurt : ",kurtosis(x_binom))



#%%

import numpy as np
from scipy.stats import kurtosis, skew

x_random = np.random.normal(0, 2, 10000)

x = np.linspace( -5, 5, 10000 )
y = 1./(np.sqrt(2.*np.pi)) * np.exp( -.5*(x)**2  )  # normal distribution

import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(x_random, bins='auto')
ax1.set_title('probability density (random)')
ax2.hist(y, bins='auto')
ax2.set_title('(your dataset)')
plt.tight_layout()

print("mean : ", np.mean(x_random))
print("var  : ", np.var(x_random))
print("skew : ",skew(x_random))
print("kurt : ",kurtosis(x_random))


#Basic statistics on MT Cars
import pandas as pd
import numpy as np

#read data
#df = pd.read_csv('data/mtcars.csv')
from pydataset import data
mtcars = data('mtcars')
mtcars.head()
df=mtcars
df.describe()
#df.dtypes()
#data distributions for 
df.columns

#%%% =========================================
# #Skewness: It represents the shape of the distribution.
#Skewness can be quantified to define the extent to which a distribution differs from a normal distribution.
#For calculating skewness by using df.skew() python inbuilt function.

df.mpg
df.mpg.skew()  #positive : right skewed, moderate, right tail longer
#majority of values in left of mean
#If skewness is not close to zero, then your data set is not normally distributed.If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.  If skewness is less than -1 or greater than 1, the distribution is highly skewed. If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.

df.mpg.plot(kind='density')

#%%%
#Kurtosis: Kurtosis is the measure of thickness or heaviness of the given distribution.#Its actually represents the 
#height of the distribution.
#The distribution with kurtosis equal to 3 is known as mesokurtic. 
#A random variable which follows normal distribution has kurtosis 3.
#If the kurtosis is less than three, the distribution is called as platykurtic. 
#Here,the distribution has shorter and thinner tails than normal distribution.
#If the kurtosis is greater than three, the distribution is called as leptykurtic. 
#Here, the distribution has longer and fatter tails than normal distribution.
#For calculating kurtosis by using df.kurtosis() python inbuilt function.
# ========
#Baseline: Kurtosis value of 0

#Data that follow a normal distribution perfectly have a kurtosis value of 0. 
#Normally distributed data establishes the baseline for kurtosis. 
#Sample kurtosis that significantly deviates from 0 may indicate that the data are not normally distributed.


#meso kurtic : +3 between 
#https://www.quora.com/What-does-a-negative-kurtosis-indicates   : Read this

df.mpg.kurtosis()  # towards plateau away from normal
df.mpg.plot(kind='density')



#All columns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1) # matrix of 2 x 2 plots : first plot
df["mpg"].plot.kde() 
plt.title('Mileage')
plt.subplot(2, 2, 2) # matrix of 2 x 2 plots : 2nd plot
df.wt.plot.kde() 
plt.title('Weight')
plt.subplot(2, 2, 3) # matrix of 2 x 2 plots : 3nd plot
df.hp.plot.kde() 
plt.title('Horse Power')
plt.subplot(2, 2, 4) # matrix of 2 x 2 plots : 4th plot
df["disp"].plot.kde() 
plt.title('Displacement')
plt.show();








#%%
#Topic ----Statistics - T-Test
#link : https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/
#The Student’s t-Test is a statistical hypothesis test for testing 
#whether two samples are expected to have been drawn from the same 
#population. It is named for the pseudonym “Student” used by 
#William Gosset, who developed the test.
#The test works by checking the means from two samples to see 
#if they are significantly different from each other. It does 
#this by calculating the standard error in the difference 
#between means, which can be interpreted to see how likely 
#the difference is, if the two samples have the same mean 
#(the null hypothesis).
#The t statistic calculated by the test can be interpreted 
#by comparing it to critical values from the t-distribution. 
#The critical value can be calculated using the degrees of 
#freedom and a significance level with the percent point 
#function (PPF).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% One Sample Test
#compare sample data with assumed mean
import scipy.stats as stats
#https://docs.scipy.org/doc/scipy/reference/stats.htmlam i audi



np.random.seed(1234)

population_marks = np.random.normal(loc=55, scale=12, size=10000)
population_marks

sample_marks=population_marks[400:430]
sample_marks

plt.hist(sample_marks)

np.mean(sample_marks)

assumed_mean = np.mean(population_marks)
assumed_mean

#Hypothesis Ho: mean=60, Ha: mean- <> 60 :Two tail test

ttest1S1 = stats.ttest_1samp(a=sample_marks, popmean=assumed_mean)


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html#scipy.stats.ttest_1samp
ttest1S1
#pvalue < 0.05 : hence REJECT Ho (Null Hypothesis) in favour of Ha
#ie. True mean is not equal to 60
ttest1S2 = stats.ttest_1samp(a=sample_marks, popmean=55)
ttest1S2
#now pvalue > .05 : Hence accept Ho hypothesis

#%%%
from statsmodels.stats.weightstats import ttest_ind
#Let us generate some random data from the Normal Distriubtion.
#We will sample 50 points from a normal distribution 
#with mean μ=0 and variance σ2=1 and from another with mean μ=2 and variance σ2=1.

marks_before = np.random.normal(loc=50, scale=10, size=100)
#after undergoing training
marks_after = np.random.normal(loc=55, scale=12, size=100)
#make test
marks_before
marks_after

ttest = ttest_ind(marks_before, marks_after)

ttest
#t-statistics, p-values, degrees of freedom
ttest[0]
ttest[1] < 0.5 #if true, Null Hypothesis true that mean of both the samples is same
#under confidence interval 95%
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
from scipy.stats import ttest_ind
ttest_sp = ttest_ind(marks_before, marks_after, axis=0, equal_var=True)
ttest_sp
#same as from statsmodel : no degree of freedom

#%%%
#A One Sample T-Test is a statistical test used to evaluate the null hypothesis that the mean m of a 1D sample dataset of independant observations is equal to the true mean μ of the population from which the data is sampled. In other words, our null hypothesis is that :m=μ


#%%% Links
#https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f
#https://www.youtube.com/watch?v=pTmLQvMM-1M


#%%

import pandas as pd
# load data file
d = pd.read_csv("https://reneshbedre.github.io/assets/posts/anova/onewayanova.txt", sep="\t")

d

# generate a boxplot to see the data distribution by treatments. Using boxplot, we can easily detect the differences 
# between different treatments
d.boxplot(column=['A', 'B', 'C', 'D'], grid=False)

# load packages


import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns F and P-value
fvalue, pvalue = stats.f_oneway(d['A'], d['B'], d['C'], d['D'])
print(fvalue, pvalue)
# 17.492810457516338 2.639241146210922e-05



# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols
# reshape the d dataframe suitable for statsmodels package 
d_melt = pd.melt(d.reset_index(), id_vars=['index'], value_vars=['A', 'B', 'C', 'D'])

d_melt

# replace column names
d_melt.columns = ['index', 'treatments', 'value']




# Ordinary Least Squares (OLS) model

model = ols('value ~ index', data=d_melt).fit()
model

anova_table = sm.stats.anova_lm(model, typ=2)
anova_table






























