import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

#save a path to variable
houses_file_path = '/Users/dariavasylieva/PycharmProjects/TrainingGround/melbourneHousing.csv'
#read data and store them in DataFrame
melbourne_data = pd.read_csv(houses_file_path)
#print summary of the data
print("Summary of the data:")
print(melbourne_data.describe())
print('--------------------------------------------------------------------------------------')
#select data for modelling
print("Select Data for modelling:")
print(melbourne_data.columns)
print('--------------------------------------------------------------------------------------')
#select Prediction Target
print("Select Prediction Target:")
y = melbourne_data.Price
print(y)
print('--------------------------------------------------------------------------------------')
#select Features
print("Select Features:")
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]
print(X.describe())
print('--------------------------------------------------------------------------------------')
#print a few first rows
print("Print a few first rows only:")
print(X.head())
print('--------------------------------------------------------------------------------------')
#Split data in two for training and evaluation
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model using the part of data
melbourne_model.fit(train_X, train_y)
#See predicted values
print("The predictions are:")
val_predictions = melbourne_model.predict(val_X)
print(val_predictions)
print('--------------------------------------------------------------------------------------')
#See absolute error. How precise our model is?
print("MAE - Mean Absolute Error:")
print("Spoiler Alert: the error is HUGE! Data which model sees for the first time discovered that our model prediction skill is shit... for now.")
print(mean_absolute_error(val_y, val_predictions))



