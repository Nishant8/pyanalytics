# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:14:07 2020

@author: Nishant Agarwal
"""

# -*- coding: utf-8 -*-
"""
Wed May  9 19:56:46 2018: Dhiraj
"""
#http://pbpython.com/market-basket-analysis.html
#pip install mlxtend : in anconda prompt as admin
#https://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv('C:/Users/Nishant Agarwal/Downloads/XIM-Batch-2-master (1)/XIM-Batch-2-master/29_Apriori/online_store.csv')
df.head()
df.describe()
df.shape



#%%
#remove extra spaces from descriptions
#remove rows with empty invoices and containing C
df['Description'].head()
'''
The str. strip() function is used to remove leading and trailing characters. 
Strip whitespaces (including newlines) or a set of specified characters
from each string in the Series/Index from left and right sides.
'''
df['Description'] = df['Description'].str.strip()
df['Description'].head()

df.InvoiceNo.head()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True) #remove NA invoice nos

df['InvoiceNo'] = df['InvoiceNo'].astype('str')

'''
C is Discounted
'''

df = df[~df['InvoiceNo'].str.contains('C')]
df
#%%

#subset for France. reformat it : Invoice No and items

df1=df[1:20]
df1


basket = (df[df['Country'] =="United Kingdom"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


pd.set_option('display.max_columns', None)
basket.head(1)
#%%
#recode : 0 (nil) or 1 (> 1)
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

basket_sets.head()

print (basket_sets.values)

print(basket_sets.iloc[:, 5:8].values)

basket_sets.columns

basket_sets.drop('POSTAGE', inplace=True, axis=1) #remove POSTAGE column not an item



basket_sets.to_csv('fittedbasket.csv')

#%%
'''
import pandas as pd

basket_sets1=basket_sets[1:1000]

'''
frequent_itemsets = apriori(basket_sets, min_support=0.015, use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)
rules.head()



#%%
rules[ (rules['lift'] >= 6) &  (rules['confidence'] >= 0.8) ]

basket['ALARM CLOCK BAKELIKE GREEN'].sum()

basket['ALARM CLOCK BAKELIKE RED'].sum()

#%%
basket2 = (df[df['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket2

basket_sets2 = basket2.applymap(encode_units)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

rules2[ (rules2['lift'] >= 4) &   (rules2['confidence'] >= 0.5)]
