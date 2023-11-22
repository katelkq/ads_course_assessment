# This file contains code for suporting addressing questions in the data

import yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from functools import partial

from . import access
from . import assess

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

# parameters
conn = None
df = None
local = None


def _initialize():
    """
    Initialize required connections and parameters.
    """
    yes = {'Y', 'y', 'yes'}
    no = {'N', 'n', 'no'}

    global local
    global conn

    while True:
        choice = input('Is this a local runtime? (Y/n)').lower()
        if choice in yes:
            local = True
            
        elif choice in no:
            local = False
            print('Initializing connection to database...')

            database_details = {"url": "database-ads-kl559.cgrre17yxw11.eu-west-2.rds.amazonaws.com",
                "port": 3306}

            try:
                with open("credentials.yaml") as file:
                    credentials = yaml.safe_load(file)
                    username = credentials["username"]
                    password = credentials["password"]
                    url = database_details["url"]
            except IOError:
                print('File credentials.yaml not found. Please put it in the current directory.')
                return -1

            conn = access.create_connection(user=credentials["username"],
                                            password=credentials["password"],
                                            host=database_details["url"],
                                            database="property_prices")
            
            access.initialize_database(conn)
            
        else:
            print('Invalid response. Try again!')
            return -1

    return 0
    pass


def bbox(point1, point2, dist):
    """
    Returns whether point2 is within some distance N, S, E, W of point1.
    :param point1: a (lat, lon) tuple
    :param point2: a (lat, lon) tuple
    :param dist: distance specified in meters
    """
    # convert meters to degrees
    dist = dist / 1000 / (40075/360)
    return (np.abs(point2[0]-point1[0]) <= dist) & (np.abs(point2[1]-point1[1]) <= dist)
    pass


def split_dataset(dataset, split=0.8):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    cutoff = int(split*len(dataset))
    train = dataset.iloc[indices[:cutoff]]
    val = dataset.iloc[indices[cutoff:]]

    return train, val
    pass


def retrieve_backing_set(row, dist, timedelta):
    global df

    # retrieve the backing set of the specified row

    conditions = []
    conditions.append(bbox((row['latitude'], row['longitude']), (df['latitude'], df['longitude']), dist))
    conditions.append((df['date_of_transfer'] - row['date_of_transfer']).apply(lambda x: np.abs(x.days)) <= timedelta)
    conditions.append(df['property_type'] == row['property_type'])
    
    # conditions.append(df['new_build_flag'] == row['new_build_flag'])
    # conditions.append(df['tenure_type'] == row['tenure_type'])

    filter = np.ones(shape=len(df), dtype=bool)
    
    for condition in conditions:
        filter = filter & condition

    return df.loc[filter]

    pass


def build_feature(row, dist, timedelta):
    backing_set = retrieve_backing_set(row, dist, timedelta)

    feature = [np.mean(backing_set['price']),
                np.max(backing_set['price']),
                np.min(backing_set['price'])]
    
    list_of_tags = [{'amenity': True},
                    {'leisure': True},
                    {'shop': True}]
    
    for tags in list_of_tags:
        pois = access.retrieve_pois(place_name=f"{row['town_city']}, {row['county']}", 
                                    latitude=row['latitude'], 
                                    longitude=row['longitude'], 
                                    tags=tags, 
                                    dist=dist)
        feature.append(len(pois))

    return feature
    pass

def predict_price(latitude, longitude, date, property_type):
    """
    Price prediction for UK housing.
    """

    result = _initialize()

    if result != 0:
        print('Initialization failed.')
        return

    print('Initialization successful.')

    global df
    
    # converting date to datetime format
    date = pd.to_datetime(date)
    
    # specify time range of data to retrieve
    start, end = date.year - 2, date.year + 3

    # load relevant df
    if local:
        df_by_year = []
        for year in range(start, end):
            filepath = f'./data/pc-{year}.csv'
            df_by_year.append(assess.data(filepath=filepath))
        df = pd.concat(df_by_year, axis=0)
    else:
        df = assess.data(conn=conn, where=f"WHERE `date_of_transfer` >= '{start}-01-01' AND `date_of_transfer` <= '{end}-12-31'", local=False)

    # filter df based on property type
    df = df.loc[df['property_type'] == property_type]

    # filter again based on location
    dataset = df.loc[bbox((latitude, longitude), (df['latitude'], df['longitude']), 500)]

    train, val = split_dataset(dataset)
    print('Training and validation sets created.')

    prices = np.array(train['price'])

    # apply feature builder function to each row
    print('Creating feature vectors from training set...')
    features = np.array(train.apply(partial(build_feature, dist=500, timedelta=365), axis=1))
    features = sm.add_constant(features)

    print('Performing linear regression...')
    model = sm.OLS(prices, features)
    results = model.fit()

    print(results.summary())

    actual_prices = np.array(val['price'])

    # iterate through the validation set, making prediction for each of them
    print('Creating feature vectors from validation set...')
    pred_features = np.array(val.apply(partial(build_feature, dist=500, timedelta=365), axis=1))
    pred_prices = np.array(results.get_prediction(pred_features).summary_frame(alpha=0.05)['mean'])

    r2 = r2_score(actual_prices, pred_prices)
    print(f'R2 = {r2}')

    if r2 < 0.4:
        print('WARNING: low R2 value')

    test = {'latitude': latitude,
            'longitude': longitude,
            'date_of_transfer': date,
            'property_type': property_type}
    test_feature = build_feature(test, dist=500, timedelta=365)

    test_price = results.get_prediction(test_feature).summary_frame(alpha=0.05)['mean'][0]
    print(f'Predicted price: £{test_price}')
    return test_price
    pass
