# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:11:23 2020

@author: Nishant Agarwal
"""

#Topic: AR - efficient apriori
#-----------------------------
#libraries

#https://en.wikipedia.org/wiki/Apriori_algorithm
#https://en.wikipedia.org/wiki/Association_rule_learning


from efficient_apriori import apriori

transactions = [('eggs', 'bacon', 'soup'),    ('eggs', 'bacon', 'apple'), ('soup', 'bacon', 'banana')]
transactions

itemsets, rules = apriori(transactions, min_support=0.5,  min_confidence=0.5)

print(rules)  # [{eggs} -> {bacon}, {soup} -> {bacon}]

rules_rhs = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules)

print(rules_rhs)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule) # Prints the rule and its confidence, support, lift, ...




#%%
#Topic ---- Association Rule Analysis 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time
import logging
#%%%

'''
m -> b = 1
m = 2

(m->b) divided by (m) = 1/2 =0.5


b -> m =1

b=1

(b->m) divided by (b) =1/1 =1

w -> m = 1

w =1

(w-> m) divided by (w) = 1

m -> w = 1
m= 2

(m->w) divided by (m) =1/2 = 0.5 

'''



transactions = [['milk', 'water'], ['milk', 'bread'], ['milk','bread','water']]
transactions
#%%%
support_threshold = 0.0001

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
te_ary

'''

 bread  milk  water
 [False,  True,  True]
 [ True,  True, False]
 [ True,  True,  True]

'''
import pandas as pd

df = pd.DataFrame(te_ary, columns=te.columns_)
df


# apriori
frequent_itemsets = apriori(df, min_support=support_threshold, use_colnames = True)
frequent_itemsets
# end time to calculation#%%%

print(frequent_itemsets) #dataframe with the itemsets
pd.set_option('display.max_columns',None)


confidence = association_rules(frequent_itemsets, metric="confidence", min_threshold=0)
print(confidence) #dataframe with confidence, lift, conviction and leverage metrics calculated

print(confidence[['antecedents', 'consequents', 'confidence']])



support = association_rules(frequent_itemsets, metric="support", min_threshold = 0)
print(support)
print(support[['antecedents', 'consequents', 'support','confidence']])

'''
In Table 1, the lift of {apple -> beer} is 1,which implies no association 
between items. A lift value greater than 1 means that item Y is likely to 
be bought if item X is bought, while a value less than 1 means that item Y 
is unlikely to be bought if item X is bought.
'''

lift = association_rules(frequent_itemsets, metric="lift", min_threshold = 0)
print (lift[['antecedents', 'consequents', 'support','confidence','lift']])
