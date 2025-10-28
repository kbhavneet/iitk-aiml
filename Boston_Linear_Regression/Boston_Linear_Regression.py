# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 16:49:59 2025

@author: Dell
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd

BosData = pd.read_csv('BostonHousing.csv')
X=BosData.iloc[:,0:11]
y=BosData.iloc[:,13]
Xtrain, Xtest, ytrain, ytest=train_test_split(X,y,test_size=0.2,random_state=10)
reg=LinearRegression()
reg.fit(Xtrain,ytrain)
ytrainpredict=reg.predict(Xtrain)
mse=mean_squared_error(ytrain,ytrainpredict);
r2=r2_score(ytrain,ytrainpredict)
print('Train MSE=', mse)
print('Train R2 score=', r2)
print("\n")

ytestpredict=reg.predict(Xtest)
mse=mean_squared_error(ytest,ytestpredict)
r2=r2_score(ytest,ytestpredict)
print('Train MSE=', mse)
print('Train R2 score=', r2)

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(ytest,ytestpredict,color='blue',alpha=0.6)
plt.plot([ytest.min(),ytest.max()],[ytest.min(),ytest.max()],color='red',linestyle='--')
plt.title('Actual vs Predicted values')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.grid()
plt.show()