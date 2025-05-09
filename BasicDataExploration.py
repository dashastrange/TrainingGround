import pandas as pd
import numpy as np

#save path to variable
houses_file_path = '/Users/dariavasylieva/PycharmProjects/TrainingGround/melbourneHousing.csv'
#read data and store them in DataFrame
melbourne_data = pd.read_csv(houses_file_path)
#print summary of the data
print(melbourne_data.describe())
print('--------------------------------------------------------------------------------------')
#select data for modelling
print(melbourne_data.columns)
print('--------------------------------------------------------------------------------------')
#select Prediction Target
y = melbourne_data.Price
print(y)
print('--------------------------------------------------------------------------------------')
#select Features
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
x = melbourne_data[melbourne_features]
print(x)
print('--------------------------------------------------------------------------------------')

