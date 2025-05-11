import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

#save a path to variable
houses_file_path = '/Users/dariavasylieva/PycharmProjects/TrainingGround/Housing.csv'
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
y = melbourne_data.price
print(y)
print('--------------------------------------------------------------------------------------')
#select Features
print("Select Features:")
melbourne_features = ["area", "bathrooms", "bedrooms", "parking"]
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
print("MAE - Mean Absolute Error. val_y is data not used for trainig - it is data for reference:")
print(val_y)
print("Spoiler Alert: the error is HUGE! Data which model sees for the first time discovered that our model prediction skill is shit... for now.")
print(mean_absolute_error(val_y, val_predictions))

#Improve a model by determining the best number of nodes so we find spot between overfitting and underfitting
print("Function to compare MAE scores from different values for max_leaf_nodes:")
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae
print('--------------------------------------------------------------------------------------')
#Try to find the best number of nodes so error is smaller
print("Get MAE for different values for nodes:")
for max_leaf_nodes in [2, 5, 10, 20, 100]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
#Using 5 nodes for the model decision tree since this value of nodes gives the smallest error value
print("Now model will use 10 nodes:")
better_model = DecisionTreeRegressor(max_leaf_nodes=10, random_state=1)
better_model.fit(train_X, train_y)
predict_values = better_model.predict(val_X)
lower_mae = mean_absolute_error(val_y, predict_values)
print("The lowest achieved MAE:")
print(lower_mae)
print('--------------------------------------------------------------------------------------')




