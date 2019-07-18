import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split

file="C:\\Users\dont_do_again\Desktop\mldata\Concrete_Data.csv"
df=pd.read_csv(file)
#print(df.head())
#print(df.columns)
#print(df.isnull().sum())

x=df.drop('Concrete_compressive_strength',axis=1)
y=df['Concrete_compressive_strength']
#print(x.head())
#print(y.head())

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=0)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#make model
linreg=linear_model.LinearRegression()
linreg.fit(x_train,y_train)

w1=linreg.coef_
w0=linreg.intercept_

print("coefficients : \n",w1)
print("intercept : ",w0)

#prediction
y_pred=linreg.predict(x_test)

#calculate error
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)

print("mean absolute error is : ",mae)
print("mean squared error is : ",mse)

#checking on a new value
newx=[[535,0.0,0.0,164,2.5,1052,676.0,27]]
newy=linreg.predict(newx)
print(newy)



