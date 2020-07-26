#! python3
"""
Created on Fri Feb 14 21:31:33 2020

ECDF to show distribution and variance of desired transaction data

@author: HM
"""

import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

os.chdir('FILE PATH HERE')

table = pd.read_csv('FILE NAME HERE')

category_1 = table[table['Category'] == 'category_1']['$ Amount'].values
category_2 = table[table['Category'] == 'category_2']['$ Amount'].values
category_3  = table[table['Category'] == 'category_3']['$ Amount'].values

def ecdf(data):
    
    x = np.sort(data)
    y = np.arange(1, len(data)+1) / len(data)
    
    return x, y

x_category_1, y_category_1 = ecdf(category_1)
x_category_2, y_category_2 = ecdf(category_2)
x_category_3, y_category_3 = ecdf(category_3)

_ = plt.plot(x_category_1, y_category_1, marker='.', linestyle='none')
_ = plt.plot(x_category_2, y_category_2, marker='.', linestyle='none')
_ = plt.plot(x_category_3, y_category_3, marker='.', linestyle='none')
_ = plt.xlabel('$ Amount Spent')
_ = plt.ylabel('ECDF')
_ = plt.legend(['category_1', 'category_2', 'category_3'], loc='lower right')
plt.show()