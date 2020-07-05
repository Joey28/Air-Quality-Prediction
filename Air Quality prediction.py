# Air Quality Prediction using Multiple Linear Regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
train= pd.read_csv('Train.csv')
X_test= pd.read_csv('Test.csv')

X_train= train.iloc[:, :-1].values
y_train= train.iloc[:, 5].values

# building the multiple linear regression model
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

#predicting for the test data
y_pred= regressor.predict(X_test)
