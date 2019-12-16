#Simple Linear Regression

#Data Preprocessing#
#Importing the libraries

import numpy as np #for importing mathematical operations
import matplotlib.pyplot as plt #for importing plots and figures
import pandas as pd #for importing files 

#Importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values #
y = dataset.iloc[:,1].values

#Splitting data into Training Data and Test Data
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#Feature_Scalinng
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

#Fitting Simple Lniear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(x_test)

#Visualizing the training set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#Visualizing the test set results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

