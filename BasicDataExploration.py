import pandas as pd
import numpy as np

cases_file_path = '/Users/dariavasylieva/PycharmProjects/TrainingGround/TestCasePriority.csv'
cases_data = pd.read_csv(cases_file_path)
#print(cases_data.describe())
#print(cases_data.columns)
print(cases_data.PriorityFromPO)