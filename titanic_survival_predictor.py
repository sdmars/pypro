import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split


#reading the csv file
file="C:\\Users\dont_do_again\Desktop\mldata\\train.csv"
dfm=pd.read_csv(file)
df=dfm.dropna(axis=0,how='any')
#print(df.isnull().sum())
#print(df.columns)


#creating dummy boolean values from categorical variables
embark=pd.get_dummies(df['Embarked'],drop_first=True)
pc=pd.get_dummies(df['Pclass'],drop_first=True)

#concat these columns and drop 2 columns which contains string values
df=pd.concat([df,embark,pc],axis=1)
df.drop('Embarked',axis=1,inplace=True)
df.drop('Pclass',axis=1,inplace=True)

#check still a string exists or not
#print(df.apply(lambda row: row.astype(str).str.contains('Test').any(),axis=1))
#print(df.columns)
print(df.head())
x=df.drop('Survived',axis=1)
y=df['Survived']


#train test splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#print(x_test)
#print(y_test)

#model initialization
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)

#y_test.shape=(-1,1)
#predit y
y_pred=logmodel.predict(x_test)

#confusion_matrix
cfm=confusion_matrix(y_test,y_pred)
print(cfm)

#printing accuracy score
accs=accuracy_score(y_test,y_pred)
print(accs)


newx=[[38.0,5,0,0,1,0,0,0,0]]
newy=logmodel.predict(newx)
print(newy)






