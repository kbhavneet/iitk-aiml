import matplotlib.pyplot as plt
# import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets # we are using in-built datasets in this code
from sklearn.metrics import mean_squared_error # metric to evaluate performance of the model after training it to see how accurate they are
from sklearn.metrics import r2_score # metric to evaluate performance of the model after training it to see how accurate they are
# 3 classes 0, 1, 2: Iris setosa, Iris virginica and Iris versicolor

irisset = datasets.load_iris() # importing the dataset


#The Iris dataset was used in R.A. Fisher's classic 1936 paper, 
#The Use of Multiple Measurements in Taxonomic Problems, 
#and can also be found on the UCI Machine Learning Repository.

# It includes three iris species with 50 samples each as well as 
#some properties about each flower.
# The 3 species of iris are
#Iris setosa, Iris virginica and Iris versicolor

#The columns in this dataset are:    
#Id
#SepalLengthCm
#SepalWidthCm
#PetalLengthCm
#PetalWidthCm
#Species


X  = irisset.data[:50,0:1] # X = regressors #matrix #sepal length
y  = irisset.data[:50,1] # y = target variable , actual/original sepal width # vector n array
# matrices: capital letters, vectors: small letters
reg = LinearRegression().fit(X, y) # fitting means training (it will train the model and learn the parameters)
# regression coefficients (h0, h1...) are the parameters it will learn and NOT X. 1 feature(regressor) so 2 coefficients: h0 is bias ie regression.intercept and h1 is for sepal length 
yPredict = reg.predict(X) # predicted sepal width after training # vector
mse = mean_squared_error(y, yPredict) # score/value
r2 = r2_score(y, yPredict) # score/value # recursion score btw 0 and 1. Closer to 1 means better model # goodness of fit

print('Train MSE =', mse)
print('Train R2 score =', r2)
print("\n")

plt.figure() #starting a new figure
plt.scatter(y, yPredict, color='blue', alpha=0.6) # we are using a scatter plot # 50 entries in y and 50 entries in yPredict so their corresponding entries are giving you coordinates. 
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Values') 
plt.xlabel('Actual SepalWidth')
plt.ylabel('Predicted SepalWidth')
plt.grid() # for 2-D grid
plt.show()

# NOTES
# If model is ideal, these points y and yPredict should lie on red straight
# line (representing x = y ie straight line) and y = yPredict

# train_data = np.hstack((X, y.reshape(-1, 1)))
# print("Training Data (Sepal Length, Sepal Width):")
# print(train_data)