# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 20:48:28 2020

@author: Nishant Agarwal
"""

#Topic ---- MB - store
#https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
#%%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from efficient_apriori import apriori

st_data = pd.read_csv('C://Users//Nishant Agarwal//Downloads//XIM-Batch-2-master (1)//XIM-Batch-2-master//29_Apriori//store_data1.csv',')
store_data.shape
store_data = pd.read_csv('C://Users//Nishant Agarwal//Downloads//XIM-Batch-2-master (1)//XIM-Batch-2-master//29_Apriori//store_data1.csv', header=None)
store_data.head()ore
#%%%
'''
aa=store_data.to_numpy()
aa.shape
'''

store_data.shape


records = []

for i in range(0, 7501):
    print(i)
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])

records




association_rules = apriori(records, min_support=0.05)
association_rules
association_results = list(association_rules)
association_results


te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
te_ary.shape


df = pd.DataFrame(te_ary, columns=te.columns_)
df['nan']

support_threshold=0.001

frequent_itemsets = apriori(df, min_support=support_threshold, use_colnames = True)




confidence = association_rules(frequent_itemsets, metric="confidence", min_threshold=0)
print(confidence) #dataframe with confidence, lift, conviction and leverage metrics calculated
print(confidence[['antecedents', 'consequents', 'support','confidence']])

support = association_rules(frequent_itemsets, metric="support", min_threshold = 0)
print(support)
print(support[['antecedents', 'consequents', 'support','confidence']])

lift = association_rules(frequent_itemsets, metric="lift", min_threshold = 0)
a=lift[['antecedents', 'consequents', 'support','confidence','lift']]

print (a[1:10])

a[(a.lift>12) & (a.confidence>0.5)][1:2]

