# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:31:05 2023

@author: HP
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#Reading dataset
data_set=pd.read_csv("income_data.csv")
print(data_set)

#Ploting the countplot of workclass

sns.countplot(x='workclass', data=data_set,)
plt.show()

#Plotting the boxplot 
def graph(y):
    
    sns.boxplot(x='workclass', y=y, data=data_set)
    plt.figure(figsize=(13, 15))
    
graph('age')
graph('hours-per-week')
graph('educational-num')

plt.show()

#plotting heatmap
sns.heatmap(data_set.corr(method='pearson').drop(['age'],axis=1).drop(['age'],axis=0),annot=True);


#violin plot
fig, ax= plt.subplots(figsize = (9, 7))
sns.violinplot(ax = ax , x = data_set["age"],y=data_set["workclass"])

#plotting pairplot
sns.pairplot(data_set.drop(['workclass'], axis = 1),
             hue = 'education', height=2)
plt.show()

#DecisionTree 

x= data_set.iloc[:, 0:1].values
y= data_set.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#fitting K-NN classifier to the training set
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=5, random_state=1)

classifier.fit(x_train, y_train)
#predicting the test set result
y_pred= classifier.predict(x_test)
result=classifier.score(x_test, y_test)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
disp= ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print("Classification Report:\n" ,classification_report(y_test, y_pred))

#print(cm)

acs=accuracy_score(y_test, y_pred)
print("Accuracy =" ,acs)