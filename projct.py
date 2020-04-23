import numpy as np
import pandas as pd
import matplotlib as plt

#Preprocessing the traing set
dataset = pd.read_csv('train.csv')

dataset = dataset.drop(['PassengerId','Name','Cabin','Ticket'],axis = 1)

#encoding
dataset = pd.get_dummies(dataset, columns = ['Embarked'],drop_first = True)
dataset = pd.get_dummies(dataset, columns = ['Pclass'],drop_first = True)

dataset.columns.values

arr= [ 'Pclass_2', 'Pclass_3', 'Sex', 'SibSp', 'Parch', 'Embarked_Q',
       'Embarked_S','Age', 'Fare','Survived']


dataset = dataset[arr]
dataset['Sex'] = dataset['Sex'].map({'male':1,'female':0})

dataset['Age'] =dataset['Age'].fillna(dataset['Age'].mean())

dataset['SibSp'] = dataset['SibSp'].map({0:0,1:1,2:1,3:1,4:1,5:1,8:1})

dataset['Parch'] = dataset['Parch'].map({0:0,1:1,2:1,3:1,4:1,5:1,6:1})

#Scaling
column_to_scale = ['Age','Fare']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(dataset[column_to_scale])
dataset[column_to_scale]=sc.transform(dataset[column_to_scale])


X_train = dataset.iloc[:,:-1].values

y_train = dataset.iloc[:,9].values

####################PREPROCESSING THE TEST SET#######################################

dataset1 = pd.read_csv('test.csv')
df = dataset1.copy()
df.columns.values

dataset1 = dataset1.drop(['PassengerId','Name','Cabin','Ticket'],axis = 1)

#second method of encoding
dataset1 = pd.get_dummies(dataset1, columns = ['Embarked'],drop_first = True)
dataset1 = pd.get_dummies(dataset1, columns = ['Pclass'],drop_first = True)

dataset1.columns.values

arr= [ 'Pclass_2', 'Pclass_3', 'Sex', 'SibSp', 'Parch', 'Embarked_Q',
       'Embarked_S','Age', 'Fare']


dataset1 = dataset1[arr]
dataset1['Sex'] = dataset1['Sex'].map({'male':1,'female':0})

dataset1['Age'] =dataset1['Age'].fillna(dataset1['Age'].mean())

dataset1['Fare'] =dataset1['Fare'].fillna(dataset1['Fare'].mean())

dataset1['Parch'] =dataset1['Parch'].fillna(dataset1['Parch'].median())

dataset1['SibSp'] = dataset1['SibSp'].map({0:0,1:1,2:1,3:1,4:1,5:1,8:1})

dataset1['Parch'] = dataset1['Parch'].map({0:0,1:1,2:1,3:1,4:1,5:1,6:1,9:1})

#dataset1['Parch'].unique()
#Scaling
column_to_scale1 = ['Age','Fare']


sc.fit_transform(dataset1[column_to_scale1])
dataset1[column_to_scale1]=sc.transform(dataset1[column_to_scale1])

X_test = dataset1.iloc[:,0:9].values

#Outling the model

from sklearn.svm import SVC
classifier=SVC(kernel= 'rbf',random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)


df = df.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked'],axis = 1)

df['Survived'] = y_pred

df1 =df.copy()

df.to_csv('Survived.csv',index =False)