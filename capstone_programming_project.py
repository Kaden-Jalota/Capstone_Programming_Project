# *********************************************
# Author: 3276045
# Assessment: Capstone programming project
# Date: 13/12/24
# *********************************************

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# Code adapted from:
# https://www.geeksforgeeks.org/convert-excel-to-csv-in-python/
# Retrieved on 5/12/24

# Reading dataset and assigning it to 'elecData'
elecData = pd.read_csv('Electricity.csv', encoding = 'latin', low_memory = False)

# Removing duplicates
elecData = elecData.drop_duplicates()

# Filtering out cells with bad values
filtered_df = elecData[elecData['SMPEP2'] != '?']

filtered_df = filtered_df[["ForecastWindProduction", "PeriodOfDay", "SystemLoadEA", "SMPEA", "ORKTemperature", "ORKWindspeed",
        "CO2Intensity", "ActualWindProduction", "SystemLoadEP2", "SMPEP2"]]

for features in filtered_df:
    filtered_df[features] = pd.to_numeric(filtered_df[features], errors = "coerce")

# Filtering out empty cells
filtered_df.isnull().sum()


# Code adapted from:
# https://www.geeksforgeeks.org/python-pandas-dataframe-dropna/
# Retrieved on 7/12/24

filtered_df = filtered_df.dropna()
filtered_df.head()

# Diplaying remaining features sorted by pearson's correlation coefficient
filtered_df.corrwith(filtered_df["SMPEP2"]).abs().sort_values(ascending=False)


# Dropping features which have insufficient correlation to 'SMPEP2'
x = filtered_df.drop('SMPEP2', axis=1).drop('ForecastWindProduction', axis=1).drop('ORKTemperature', axis=1).drop('ORKWindspeed', axis=1).drop('CO2Intensity', axis=1).drop('ActualWindProduction', axis=1)
y = filtered_df['SMPEP2']

# Defining testing and training variables and splitting the dataset into 70% training and 30% testing
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size=0.3)
print(xTrain, xTest, yTrain, yTest)

def targetHistogram():
    filtered_df.hist(['SMPEP2'])

targetHistogram()

def predictorHistograms():
    filtered_df.hist(['SystemLoadEA', 'SystemLoadEP2', 'SMPEA', 'PeriodOfDay'], figsize=(15, 15))

predictorHistograms()


# Code adapted from:
# https://www.geeksforgeeks.org/matplotlib-pyplot-scatter-in-python/
# Retrieved on 9/12/24

def correlationScatterplots(predictor):
    plt.figure()

    x1 = filtered_df['SMPEP2']
    x2 = predictor #filtered_df[predictor]

    plt.scatter(x1, x2)
    plt.scatter(x1, x1)
    plt.plot(x1, x1)
    plt.show()

correlationScatterplots(filtered_df['SMPEA'])
correlationScatterplots(filtered_df['SystemLoadEA'])
correlationScatterplots(filtered_df['SystemLoadEP2'])
correlationScatterplots(filtered_df['PeriodOfDay'])


# Code adapted from:
# https://www.geeksforgeeks.org/python-linear-regression-using-sklearn/ ;
# https://www.geeksforgeeks.org/mean-squared-error/
# Retrieved on 10/12/24


# Testing a linear regression model
model1 = LinearRegression()
model1.fit(xTrain, yTrain)
model1Pred = model1.predict(xTest)
print(np.sqrt(mean_squared_error(yTest, model1Pred)))


# Testing a random forest regressor model
model2 = RandomForestRegressor()
model2.fit(xTrain, yTrain)
model2Pred = model2.predict(xTest)
print(np.sqrt(mean_squared_error(yTest, model2Pred)))


# Testing a decision tree regressor model
model3 = DecisionTreeRegressor()
model3.fit(xTrain, yTrain)
model3Pred = model3.predict(xTest)
print(np.sqrt(mean_squared_error(yTest, model3Pred)))


# Testing an adaboost regressor model
model4 = AdaBoostRegressor()
model4.fit(xTrain, yTrain)
model4Pred = model4.predict(xTest)
print(np.sqrt(mean_squared_error(yTest, model4Pred)))


# Testing an SVR model
model5 = SVR()
model5.fit(xTrain, yTrain)
model5Pred = model5.predict(xTest)
print(np.sqrt(mean_squared_error(yTest, model5Pred)))


# # Code adapted from:
# https://www.youtube.com/watch?v=YyvdWDIpafM
# Retrieved on 10/12/24

# Selecting and training random forest regressor as final model due to the smallest average error
finalModel = RandomForestRegressor()
finalModel.fit(x, y)

# Defining the target and predictors and exporting the final model as a file
predictors = x
target = y

joblib.dump(value = [finalModel, predictors, target], filename = 'finalModel.pkl')