# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:34:03 2019

@author: Alfredo
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
from config import DATA_TRAIN_ROUTE, DATA_TEST_ROUTE, DATA_TEST_FINAL_ROUTE, RESULT_ROUTE
from missingpy import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy import stats
import time
import datetime

def pre_processing(data_route):
    
    data_frame = pd.read_csv(data_route)
    #Missing Value Imputation by Random Forest
    real_colums = data_frame.columns
    def handle_column_negative(x):
        return x.map(lambda x: x * (-1) if x < 0 else x)

    numericData = data_frame.copy()
    # Preparing data to Random Forest
    numericData = numericData.drop(["cluster","date", "country"], axis = 1)
    numericData = numericData.apply(lambda x: handle_column_negative(x), axis=1)
    numericData = numericData.replace([np.inf, -np.inf], np.nan)
    
    # applying random forest
    random_forest_imputer = KNNImputer()
    random_forest_result = random_forest_imputer.fit_transform(numericData)
    data_frame_processed = pd.DataFrame(random_forest_result)
    
    # adding removed fields
    data_frame_processed.insert(0, column='date', value=data_frame['date'])
    data_frame_processed.insert(0, column='cluster', value=data_frame['cluster'])
    data_frame_processed.insert(0, column='country', value=data_frame['country'])
    
    data_frame_processed.columns = real_colums
    return data_frame_processed

def date_to_millis(d):
    return time.mktime(datetime.datetime.strptime(d, "%Y-%m-%d").timetuple())  


def get_linear_regression_model(data_frame):
    model = LinearRegression()
    
    train = data_frame.copy()
    target = pd.DataFrame(train, columns=["volume"])
    train = train.filter(target_fields, axis = 1)
    
    train['date'] = train['date'].apply(date_to_millis)

    model.fit(train, target)
    return model

def get_country_and_groups_tuples(data_frame):
    countries = set(data_frame['country'])
    products = set(data_frame['cluster'])
    result = []
    for country in countries:
        for product in products:
            result.append((country, product))
    return result

def get_group(g, key):
    if key in g.groups: return g.get_group(key)
    return pd.DataFrame()

def create_models(data_frame):
    # create a model for each country and product
    models = dict()

    train_data_frame_by_country_cluster = data_frame.groupby(['country','cluster'])
    
    groups = get_country_and_groups_tuples(data_frame)
    
    for (country, product) in groups:
        country_product_frame = get_group(train_data_frame_by_country_cluster, (country, product))
        if country_product_frame.empty != True:
            model = get_linear_regression_model(country_product_frame)
            models[(country, product)] = model
    return models

def predict(models, test_data_frame):
    # test_groups = get_country_and_groups_tuples(test_data_frame)
    test_group_by = test_data_frame.groupby(['country','cluster'])

    def predict(row):
        group = (row['country'], row['cluster'])
        test_data_frame = get_group(test_group_by, group)
        if test_data_frame.empty != True:
            test_data_frame = test_data_frame.filter(target_fields)
            test_data_frame['date'] = test_data_frame['date'].apply(date_to_millis)
            
            try:
                group_model = models[group]
                return group_model.predict(test_data_frame)
            except:
                return []
        else:
            return []
    
    predictions = dict()
    for (i, row) in test_data_frame.iterrows():
        predictions[(
                row['country'],
                row['cluster'],
                row['date'],
                )] = predict(row)
    return predictions

#OUTLIER
def outlier_treatment(train_data_frame):
    numericData = train_data_frame.loc[:,"expenses":"volume"]
    cleaned_data = numericData.copy()
    cleaned_data[~(np.abs(stats.zscore(cleaned_data)) < 3).all(axis=1)] = np.nan
    imputer = KNNImputer()
    result = imputer.fit_transform(cleaned_data)
    cdp = pd.DataFrame(result)
    cdp.insert(0, column='date', value=train_data_frame['date'])
    cdp.insert(0, column='cluster', value=train_data_frame['cluster'])
    cdp.insert(0, column='country', value=train_data_frame['country'])
    cdp.columns = train_data_frame.columns.copy()
    return cdp

def store_result(data_frame, predictions):
    result_rows = [
            "country",
            "cluster",
            "date",
            "upper_bound",
            "forecast",
            "lower_bound",
            ]
    def generate_prediction_range(predictions):
        upper_bound = np.max(predictions)
        lower_bound = np.min(predictions)
        forecast = np.mean(predictions)
        return (upper_bound, forecast, lower_bound)

    # result_data_frame = data_frame.filter(result_rows)
    result_data_frame = pd.DataFrame(columns = result_rows)
    for key, value in predictions.items() :
        row = dict()
        (upper_bound, forecast, lower_bound) = generate_prediction_range(value)
        country, product, date = key
        row['country'] = country
        row['cluster'] = product
        row['date'] = date
        row['upper_bound'] = upper_bound
        row['forecast'] = forecast
        row['lower_bound'] = lower_bound
        result_data_frame = result_data_frame.append(row, ignore_index=True)

    result_data_frame.to_csv (RESULT_ROUTE, index = None, header=True)

target_fields = [
    "date", 
    "expenses_1", 
    "expenses_2", 
    "expenses_3", 
    "expenses_4", 
    "expenses_5", 
    "expenses_6"
]
 
print('-----------------------------------------')
print('Start Preprocessing!')
print('-----------------------------------------')

train_data_frame = pre_processing(DATA_TRAIN_ROUTE)
test_data_frame = pre_processing(DATA_TEST_ROUTE)
train_data_prepared = outlier_treatment(train_data_frame)

print('-----------------------------------------')
print('Preprocessing Finished!')
print('-----------------------------------------')

models = create_models(train_data_prepared)

predictions = predict(models, test_data_frame)

store_result(test_data_frame, predictions)