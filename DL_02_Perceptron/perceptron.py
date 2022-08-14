
import numpy as np
import pandas as pd
import sklearn.datasets

"""# Loading Dataset"""

breast_cancer= sklearn.datasets.load_breast_cancer()

print(breast_cancer.data)

print(breast_cancer.target)

print(breast_cancer.target_names)

print(breast_cancer.feature_names)

data= pd.DataFrame(breast_cancer.data,columns= breast_cancer.feature_names)
data['class']= breast_cancer.target

data

data.describe()

print(data['class'].value_counts())

data.groupby('class').mean()

"""# Train_Test_Split"""

from sklearn.model_selection import train_test_split

X= data.drop('class',axis =1) #Load everything except class column(axis = 1)
Y=data['class']

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=.15, random_state=2, stratify= Y) 
#Split in 85-15 ratio
#random_state to ensure same split in every run
#stratify to avoid unequal distribution of class among test and train

print(Y.mean(), Y_train.mean(), Y_test.mean())



"""# Data preparation"""

import matplotlib.pyplot as plt

plt.plot(X_train.T,'.')
plt.xticks(rotation='vertical') #show xlabels vertically
plt.show()

X_train

from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()
X_train= pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test= pd.DataFrame(scaler.fit_transform(X_test), columns= X_test.columns)
#Standardizing feature values using min-max scaler

X_train

plt.plot(X_train.T,'.')
plt.xticks(rotation='vertical') #show xlabels vertically
plt.show()

#Pandas dataframe to Numpy arrays for faster calculations
X_train= X_train.values
X_test= X_test.values

print(type(X_train))

"""# Rough Model"""

from sklearn.metrics import accuracy_score

w=np.ones(X_train.shape[1]) #intial weights =1
b=0 #initial threshold

for i in range(100): #Iterating 100 times over data
    for x,y in zip(X_train,Y_train):
        if np.dot(x,w) < b:
            y_pred= 0
        else:
            y_pred=1
        if y == 1 and y_pred == 0:
            w=w+ x*.005 #setting learning rate .005
            b=b- 1*.005
        if y == 0 and y_pred == 1:
            w=w- x*.005
            b=b+ 1*.005

y_pred_test=[]
for x in X_test:
    if np.dot(x,w) < b:
        y_pred= 0
    else:
        y_pred=1
    y_pred_test.append(y_pred)

print(y_pred_test)

accuracy= accuracy_score(y_pred_test,Y_test)
print(accuracy)



"""# Perceptron Neuron"""

from sklearn.metrics import accuracy_score

class Perceptron:

    def __init__(self):
        self.b=None
        self.w=None
    
    def  model(self, x):
        return 1 if np.dot(x,self.w) >= self.b else 0

    def predict(self, X):
        y=[]
        for x in X:
            pred= self.model(x)
            y.append(pred)
        return np.array(y)

    def fit(self, X, Y, epochs=1, lr=1):
        best_accuracy= 0
        self.b= 0
        self.w= np.zeros(X.shape[1])
        best_b= None
        best_w= None
        for i in range(epochs):
            for x,y in zip(X,Y):
                pred= self.model(x)
                if y==1 and pred == 0:
                    self.w= self.w + x*lr
                    self.b= self.b - 1*lr
                elif y==0 and pred == 1:
                    self.w= self.w - x*lr
                    self.b= self.b + 1*lr
       
            accuracy= accuracy_score(self.predict(X),Y)
            if accuracy > best_accuracy:
                best_accuracy= accuracy
                best_b= self.b
                best_w= self.w

        self.b=best_b
        self.w= best_w

perceptron= Perceptron()
perceptron.fit(X_train,Y_train,1000,.005)

pred=perceptron.predict(X_test)

accuracy= accuracy_score(pred,Y_test)
print(accuracy)

