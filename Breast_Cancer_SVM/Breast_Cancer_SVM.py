from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#added
import matplotlib.pyplot as plt
import numpy as np


bcancer = datasets.load_breast_cancer()

#Breast Cancer Wisconsin (Diagnostic) Data Set
#Classes = 2
#Samples per class = 212(M),357(B)
# Samples total = 569
# Dimensionality = 30
# Features are real, positive

# Feature names
#'mean radius' 'mean texture' 'mean perimeter' 'mean area'
# 'mean smoothness' 'mean compactness' 'mean concavity'
# 'mean concave points' 'mean symmetry' 'mean fractal dimension'
# 'radius error' 'texture error' 'perimeter error' 'area error'
# 'smoothness error' 'compactness error' 'concavity error'
# 'concave points error' 'symmetry error' 'fractal dimension error'
# 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
# 'worst smoothness' 'worst compactness' 'worst concavity'
# 'worst concave points' 'worst symmetry' 'worst fractal dimension'




#added
X = bcancer.data[:100,:2]
#y=bcancer.target[:100]
Ytest=bcancer.target[:100]
clf=SVC(kernel='linear')
#clf.fit(X,y)
clf.fit(X,Ytest)
Ypred=clf.predict(X)
w=clf.coef_[0]







# Linear SVM



svmcscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of Linear SVM Classifier is',100*svmcscore,'%\n')

cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of SVC with Linear Kernel is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()

# Kernel SVM RBF - Gaussian Kernal



svmcscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of SVM Classifier with RBF Kernel is',100*svmcscore,'%\n')

cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of SVC with RBF Kernel is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()
#added
plt.show()

# Kernel SVM Polynomial 



svmcscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of SVM Classifier with Polynomial Kernel is',100*svmcscore,'%\n')

cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of SVC with Polynomial Kernel is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()
#added
plt.show()

# Kernel SVM Sigmoid 



svmcscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of SVM Classifier with Sigmoid Kernel is',100*svmcscore,'%\n')

cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of SVC with Sigmoid Kernel is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()
#added
plt.show()