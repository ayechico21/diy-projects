
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

print(data)

data.describe()

print(data['class'].value_counts())

data.groupby('class').mean()

"""# Train_Test_Split"""

from sklearn.model_selection import train_test_split

X= data.drop('class',axis =1) #Load everything except class column(axis = 1)
Y=data['class']

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=.15, random_state=1, stratify= Y) 
#Split in 85-15 ratio
#random_state to ensure same split in every run
#stratify to avoid unequal distribution of class among test and train

print(Y.mean(), Y_train.mean(), Y_test.mean())



"""# Binarization of input"""

import matplotlib.pyplot as plt

plt.plot(X_train.T,'.')
plt.xticks(rotation='vertical') #show xlabels vertically
plt.show()

#convert features values into Binary 
X_train= X_train.apply(pd.cut,bins=2,labels=[1,0])
X_test= X_test.apply(pd.cut,bins=2,labels=[1,0])

"""value of features are greater for class 0, so labels are 1,0

inferring higher bin values indicates class 0
"""

plt.plot(X_train.T,'.')
plt.xticks(rotation='vertical') #show xlabels vertically
plt.show()

#Pandas dataframe to Numpy arrays for faster calculations
X_train= X_train.values
X_test= X_test.values

print(type(X_train))



"""# Rough Model"""

from sklearn.metrics import accuracy_score

print(X_train)

print(X_train.shape)

scores={} #stores accuracy with value of b as key
for b in range(X_train.shape[1] +1): #b can range between 0 - no. of features
    y_pred_train=[]
    for x in X_train:
        y_pred= np.sum(x) >= b
        y_pred_train.append(y_pred)
    accuracy= accuracy_score(y_pred_train,Y_train)
    scores[b]=accuracy

print(scores)
b=max(scores,key=scores.get) #highest accuracy key i.e value of b
print(b)

y_pred_test=[]
for x in X_test:
    y_pred= np.sum(x) >= b
    y_pred_test.append(y_pred)
accuracy=accuracy_score(y_pred_test,Y_test)
print("b: ",b)
print("Accuracy: ",accuracy)



"""# MP Neuron """

class MPNeuron:

    def __init__(self):
        self.b=None

    def model(self,x):
        return np.sum(x) >=self.b

    def predict(self,X):
        y=[]
        for x in X:
            y_pred= self.model(x)
            y.append(y_pred)
        return np.array(y)

    def fit(self,X):
        scores={}
        for b in range(X.shape[1] +1):
            self.b= b
            Y_pred= self.predict(X)
            scores[b]=accuracy_score(Y_train,Y_pred)
        self.b= max(scores, key=scores.get)
        print("Value of b: ",self.b)

mp_neuron= MPNeuron()
mp_neuron.fit(X_train)

Y_pred= mp_neuron.predict(X_test)
accuracy= accuracy_score(Y_test,Y_pred)
print("Accuracy: ",accuracy)
#print(f"Accuracy: {accuracy*100:.2f}%")

