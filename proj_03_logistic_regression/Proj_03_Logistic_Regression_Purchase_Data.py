import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def sigmoid(x):
  return 1/(1 + np.exp(-x))

purchaseData = pd.read_csv('Purchase_Logistic.csv')

#Logistic-Regression-Social-Network-Ads
#Using Logistic Regression model to predict if a person is 
#going to buy a new car or not based on the available data

#Problem
#A company is planning to launch a campaign for their new car brand 
#and would like to analyze which customers are most likely to 
#purchase the car so that the ads can specifically target them 
#To achieve this, they consult a social network advertising company 
#that possesses the data from another similar campaign. 
#It is now desired to construct a model to achieve the above goal.

#Dataset
#The dataset contains 400 entries for each of the features 
#userId
#gender
#age
#estimatedsalary 

#The target is 
#purchased history 
#The features taken into account are age and estimated salary which are 
#required to predict if the user will purchase a new car (1=Yes, 0=No)

X = purchaseData.iloc[:, [2, 3]] # both 2nd and 3rd columns included ie age & estimated salary 
Y = purchaseData.iloc[:, 4] # 4th column "Purchased"

scaler = StandardScaler() # pre-processing technique to normalise both columns coz 3rd col is 
# small nos (age) and 4th big (salary in 1000s) by computing mean and subtracting it from each entry.
# then compute standard deviation and divide each entry by the SD
# purpose is that both age and salary come in the similar range and participate equally in the training.
X = scaler.fit_transform(X) # transformimg X into pre-processed X

Xtrain, Xtest, Ytrain, Ytest \
= train_test_split(X, Y, test_size = 0.20, random_state = 0)

logr = LogisticRegression()
logr.fit(Xtrain, Ytrain) # 3 coefficients h0, h1 and h2 will be learned
# purple: product not purchased, yellow: product purchased
# z=w0+w1.Age+w2.Salary & applies a sigmoid function to turn this into a probability:
# 𝑝^=1/1+𝑒^(−𝑧) (this makes sure the output is always between 0 and 1)

Ypred = logr.predict(Xtest)

plt.figure(1);  
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.suptitle('Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

col = sigmoid(np.dot(X, np.transpose(logr.coef_)) + logr.intercept_) 
# for probability of y = 1 => 1/(1 + e^-xTh)
# whatever is there inside the sigmoid function is computing h⊤x
# h0 = intercept; coefficients = h1, h2; dot product with x
# col = colour values giving the probability of y = 1 i.e. P(y = 1)
# if probabilty > 0.5 => product purchased else not purchased. (Threshold = 0.5)
cf = logr.coef_;
xplot = np.arange(-1.0,1.2,0.01); # 0.01 is step size from -1.0 to 1.2
yplot = -(cf[0,0]*xplot + logr.intercept_)/cf[0,1]

plt.figure(2);
plt.scatter(X[:, 0], X[:, 1], c = col) # colours are coming from probability distribution
plt.plot(xplot,yplot,'g')
plt.suptitle('Logistic Regression Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()
# customers on left side of line will be prdicted as 0 and on right side as 1

cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of Logistic Regression is \n',cmat,'\n')
# 58 0s (57 correctly predicted ie True Negative and 1 is falsely predicted (FP)
# and 22 1s (17 TP and 5 FN)

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()
plt.show()

LRscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of Logistic Regression is',100*LRscore,'%\n')
