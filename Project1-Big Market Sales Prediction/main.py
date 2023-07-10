# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:27:43 2023

@author: AMIT GARG
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

print(train.head())

print(train.info())

print(train.describe())

corr = train.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cbar=False)
print(corr)

plt.figure(figsize=(15, 8))
i = 1
for col in ['Outlet_Size', 'Outlet_Location_Type', 'Item_Fat_Content']:
    plt.subplot(2, 4, i)
    sns.countplot(x=col, data=train, palette='Set2')
    i += 1
    plt.subplot(2, 4, i)
    sns.countplot(x=col, hue='Outlet_Type', data=train, palette='Set2')
    i += 1
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
plt.subplot(3,2,1)
sns.kdeplot(x='Outlet_Establishment_Year',data=train,palette='Set2')
plt.subplot(3,2,2)
sns.kdeplot(x='Item_Outlet_Sales',data=train,palette='Set2')
plt.subplot(3,2,3)
sns.kdeplot(x='Item_Weight',data=train,palette='Set2')
plt.subplot(3,2,4)
sns.kdeplot(x='Item_Visibility',data=train,palette='Set2')
plt.subplot(3,2,5)
sns.kdeplot(x='Item_MRP',data=train,palette='Set2')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=train)

plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=train)
