# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 09:15:37 2018

@author: Acer
"""

import numpy as np
import pandas as pd

dataset= pd.read_csv('Churn_Modelling.csv')
X= dataset.iloc[:,3:13].values
y= dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,1]= labelencoder_X.fit_transform(X[:,1])

labelencoder_X1=LabelEncoder()
X[:,2] = labelencoder_X1.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

import xgboost
classifier = xgboost.XGBClassifier()
classifier.fit(X_train,y_train)

y_pred= classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_pred,y_test)
acc_score=accuracy_score(y_pred,y_test)

from sklearn.model_selection import cross_val_score
accuracies =cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()












