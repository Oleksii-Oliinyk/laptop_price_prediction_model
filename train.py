import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pickle 

from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.simplefilter("ignore")

import model_best_hyperparameters
import columns

# reading data
ds = pd.read_csv("train_data.csv")

# clearing data
def impute_na(df, variable, value):
    return df[variable].fillna(value)

impute_values = dict()
for column in ds.columns:
    if ds[column].dtype == 'object':
        impute_values[column] = ds[column].mode()[0]
    else:
        impute_values[column] = ds[column].mean()

for column in ds.columns:
    if ds[column].isnull().values.any():
        ds[column] = impute_na(ds, column, impute_values[column])

# encoding data  
def frequency_encoding(df, column, map_dicts):
    frequency_map = df[column].value_counts(normalize=True).to_dict()
    df[column+'_freq_encoded'] = df[column].map(frequency_map)
    map_dicts[column] = frequency_map
    return df, map_dicts

map_dicts = dict()

for column in ds.columns:
    if ds[column].dtype == 'object' and column != 'OS' and column != 'Product':
        ds, map_dicts = frequency_encoding(ds, column, map_dicts)
        
# finding anomalies
def find_skewed_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary

upper_lower_limits = dict()

for column in columns.outlier_columns:
    upper_limit, lower_limit = find_skewed_boundaries(ds, column, 4)
    upper_lower_limits[column+'_upper_limit'] = upper_limit
    upper_lower_limits[column+'_lower_limit'] = lower_limit

for column in columns.outlier_columns:
    upper_limit = upper_lower_limits[column+'_upper_limit']
    lower_limit = upper_lower_limits[column+'_lower_limit']
    ds.loc[ds[column] > upper_limit, column] = upper_limit
    ds.loc[ds[column] < lower_limit, column] = lower_limit

# scaling features
scaler = MinMaxScaler()
scaler.fit(ds[columns.scaling_columns])
X_train_scaled = scaler.transform(ds[columns.scaling_columns])
X_train = ds[columns.scaling_columns]
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
ds[columns.scaling_columns] = X_train_scaled

# save parameters 


param_dict = {'impute_values':impute_values,
              'upper_lower_limits':upper_lower_limits,
              'map_dicts':map_dicts,

             }
with open('param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# Define target and features columns
X = ds[columns.X_columns]
y = ds[columns.y_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=99)


# Building and train LGBMRegressor Model
model = LGBMRegressor(**model_best_hyperparameters.params, force_row_wise=True, verbose=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:  ", metrics.mean_absolute_error(y_test, y_pred))
print("MAPE: ", metrics.mean_absolute_percentage_error(y_test, y_pred))
print("R2:   ", metrics.r2_score(y_test, y_pred))

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
