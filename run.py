# -*- coding: utf-8 -*-
"""

@author: Anubhav Sinha
"""


# Importing ML librabries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing datasets
dataset = pd.read_excel('file_name.xls')
a = dataset.iloc[:,:1].values
b_exp = dataset.iloc[:,3].values


#splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
a_train, a_test, b_exp_train, b_exp_test =train_test_split(a,b_exp, test_size=0.9,random_state=0)


"""
a_test = a_test.fillna(0)
b_exp_test = b_exp_test.fillna(0)


# To check NaN in array and replacing it with 0
np.where(np.isnan(a_train))
np.where(np.isnan(b_exp_train))
np.nan_to_num(a_train)
np.nan_to_num(b_exp_train)

#Feature Scaling Transformation

from sklearn.preprocessing import StandardScaler
sc_a = StandardScaler();
a_train = sc_a.fit_transform(a_train)
 

sc_b = StandardScaler();
b_exp_train = b_exp_train.reshape(1,-1)
b_exp_train = sc_b.fit_transform(b_exp_train)
""" 


#Fitting Simple Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(a_train,b_exp_train)
 


#Predicting 
b_pred = regressor.predict(a_test)


#Visualizing the Training set
plt.scatter(a_train,b_exp_train,color='red')
plt.plot(a_train,regressor.predict(a_train),color='blue')
plt.title('Age vs Experience')
plt.xlabel('Age')
plt.ylabel('Experience')
plt.show()



#Visualizing Test set
plt.scatter(a_test,b_exp_test, color='red')
plt.plot(a_train,regressor.predict(a_train),color='blue')
plt.title('Age vs Experience')
plt.xlabel('Age')
plt.ylabel('Experience')
plt.show()
