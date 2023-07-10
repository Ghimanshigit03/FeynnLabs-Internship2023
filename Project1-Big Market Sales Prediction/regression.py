# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:09:36 2023

@author: AMIT GARG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import klib

df_train= pd.read_csv('train.csv')
df_test= pd.read_csv('test.csv')

print(df_train.isnull().sum())
print(df_test.isnull().sum())

df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(),inplace=True)
df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean(),inplace=True)

print(df_train.isnull().sum())

print(df_train['Outlet_Size'].value_counts())

print(df_train['Outlet_Size'].mode())

df_train['Outlet_Size'].fillna(df_train['Outlet_Size'].mode()[0],inplace=True)
df_test['Outlet_Size'].fillna(df_test['Outlet_Size'].mode()[0],inplace=True)
print(df_train.isnull().sum())
print(df_test.isnull().sum())

df_train.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
df_test.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)

klib.data_cleaning(df_train)
klib.clean_column_names(df_train)
df_train=klib.convert_datatypes(df_train)


#MODEL BUILDING
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor



le=LabelEncoder()
df_train['item_fat_content']= le.fit_transform(df_train['item_fat_content'])
df_train['item_type']= le.fit_transform(df_train['item_type'])
df_train['outlet_size']= le.fit_transform(df_train['outlet_size'])
df_train['outlet_location_type']= le.fit_transform(df_train['outlet_location_type'])
df_train['outlet_type']= le.fit_transform(df_train['outlet_type'])
X=df_train.drop('item_outlet_sales',axis=1)
Y=df_train['item_outlet_sales']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=101, test_size=0.2)

sc= StandardScaler()
X_train_std= sc.fit_transform(X_train)
X_test_std= sc.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_std, Y_train)
Y_pred_lr = lr.predict(X_test_std)
print("Linear Regression:")
print("R-squared:", r2_score(Y_test, Y_pred_lr))
print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred_lr))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, Y_pred_lr)))
print()

rf = RandomForestRegressor(n_estimators=1000)
rf.fit(X_train_std, Y_train)
Y_pred_rf = rf.predict(X_test_std)
print("Random Forest Regression:")
print("R-squared:", r2_score(Y_test, Y_pred_rf))
print("Mean Absolute Error:", mean_absolute_error(Y_test,Y_pred_rf))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(Y_test,Y_pred_rf)))
print()

svr = SVR()
svr.fit(X_train_std, Y_train)
Y_pred_svr = svr.predict(X_test_std)
print("Support Vector Regression (SVR):")
print("R-squared:", r2_score(Y_test, Y_pred_svr))
print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred_svr))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, Y_pred_svr)))
print()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_std, Y_train)
Y_pred_ridge = ridge.predict(X_test_std)
print("Ridge Regression:")
print("R-squared:", r2_score(Y_test, Y_pred_ridge))
print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred_ridge))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, Y_pred_ridge)))
print()

xgb = XGBRegressor()
xgb.fit(X_train_std, Y_train)
Y_pred_xgb = xgb.predict(X_test_std)
print("XGBoost Regression:")
print("R-squared:", r2_score(Y_test, Y_pred_xgb))
print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred_xgb))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Y_test, Y_pred_xgb)))
print()

#GRAPHICAL OBSERVATIONS
algorithms = ['Linear Regression', 'Random Forest', 'SVR', 'Ridge', 'XGBoost']
r_squared_values = [r2_score(Y_test, Y_pred_lr), r2_score(Y_test, Y_pred_rf),
                    r2_score(Y_test, Y_pred_svr), r2_score(Y_test, Y_pred_ridge),
                    r2_score(Y_test, Y_pred_xgb)]
# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(algorithms, r_squared_values)
plt.xlabel('Regression Algorithm')
plt.ylabel('R-squared')
plt.title('Comparison of Regression Algorithms (R-squared)')
plt.xticks(rotation=45)
plt.show()


mse_values = [mean_absolute_error(Y_test, Y_pred_lr), mean_absolute_error(Y_test, Y_pred_rf),
                    mean_absolute_error(Y_test, Y_pred_svr), mean_absolute_error(Y_test, Y_pred_ridge),
                    mean_absolute_error(Y_test, Y_pred_xgb)]
# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(algorithms, mse_values,color='pink')
plt.xlabel('Regression Algorithm')
plt.ylabel('Mean Absolute error')
plt.title('Comparison of Regression Algorithms (Mean Absolute Error)')
plt.xticks(rotation=45)
plt.show()


rmse_values = [np.sqrt(mean_squared_error(Y_test, Y_pred_lr)), np.sqrt(mean_squared_error(Y_test, Y_pred_rf)),
                    np.sqrt(mean_squared_error(Y_test, Y_pred_svr)), np.sqrt(mean_squared_error(Y_test, Y_pred_ridge)),
                    np.sqrt(mean_squared_error(Y_test, Y_pred_xgb))]
# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(algorithms, r_squared_values,color='green')
plt.xlabel('Regression Algorithm')
plt.ylabel('Root Mean Squared Error')
plt.title('Comparison of Regression Algorithms (Root mean square)')
plt.xticks(rotation=45)
plt.show()
