# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Hi")
# importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

# Spliting the dataset into Training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting Simmple Linear Regression for training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # empty object
regressor.fit(X_train,Y_train) # this fits a equation for the data that is
# this makes the regressor object learn from the train set. Learns the correlation between
# X_TRAIN and Y_TRAin 

#Predicting the test set results
Y_pred = regressor.predict(X_test)

# Visualising the training set results
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.xlabel('Salary')
plt.show()

plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.xlabel('Salary')
plt.show()