# AREEBA MUNIR
# ASSIGNMENT 2
# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
H = dataset.iloc[:, 2:3].values
B = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
H_train, H_test, B_train, B_test = train_test_split(H, B, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(H_train, B_train)

# Predicting the Test set results
B_pred = regressor.predict(H_test)

# Visualising the Training set results
plt.scatter(H_train, B_train, color = 'brown')
plt.plot(H_train, regressor.predict(H_train), color = 'blue')
plt.title('Head Size vs Brain Weight (Training set)')
plt.xlabel('Head Size (cm^3)')
plt.ylabel('Brain Weight (grams)')
plt.show()

# Visualising the Test set results
plt.scatter(H_test, B_test, color = 'brown')
plt.plot(H_train, regressor.predict(H_train), color = 'blue')
plt.title('Head Size vs Brain Weight (Test set)')
plt.xlabel('Head Size (cm^3)')
plt.ylabel('Brain Weight (grams)')
plt.show()


