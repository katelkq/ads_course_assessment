# This file contains code for suporting addressing questions in the data

import yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from functools import partial

from . import access
from . import assess

import logging

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
        # choice = input('Do you have the ./data directory? (Y/n) ').lower()
        choice = 'Y'

        if choice in yes:
            local = True
            break

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
            break

        else:
            print('Invalid response. Try again!')

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

    # toggle
    feature = [np.mean(backing_set['price']),
               np.max(backing_set['price']),
               np.min(backing_set['price'])]
    
    tags = {'amenity': True,
            'leisure': True,
            'shop': True}
    
    pois = access.retrieve_pois(latitude=row['latitude'], 
                                longitude=row['longitude'], 
                                tags=tags, 
                                dist=dist)
    
    if pois is None:
        feature.append(0)
        feature.append(0)
        feature.append(0)
        return feature

    for key in tags.keys():
        if key in pois.columns:
            feature.append(len(pois[key].dropna()))
        else:
            feature.append(0)

    return feature
    pass


def predict_price_split(latitude, longitude, date, property_type):
    """
    Price prediction for UK housing.
    """

    # initialization
    result = _initialize()

    if result != 0:
        print('Initialization failed.')
        return

    print('Initialization successful.')

    # converting date to datetime format
    date = pd.to_datetime(date)
    
    # specify time range of data to retrieve
    start, end = max(1995, date.year - 2), min(2021, date.year + 3)

    global df

    # load relevant df
    if local:
        df_by_year = []
        for year in range(start, end):
            filepath = f'./data/pc-{year}.csv'
            df_by_year.append(assess.data(filepath=filepath))
        df = pd.concat(df_by_year, axis=0)
    else:
        df = assess.data(conn=conn, where=f"WHERE `date_of_transfer` >= '{start}-01-01' AND `date_of_transfer` <= '{end}-12-31'", local=False)

    # filter df based on location and property type
    df = df.loc[bbox((latitude, longitude), (df['latitude'], df['longitude']), 5000)]
    df = df.loc[df['property_type'] == property_type]

    # create dataset based on time and location
    dataset_recent = df.loc[(df['date_of_transfer'] - date).apply(lambda x: np.abs(x.days)) <= 365]

    # TODO: iteratively widen range if not enough data points?
    dist = 500
    dataset = dataset_recent.loc[bbox((latitude, longitude), (dataset_recent['latitude'], dataset_recent['longitude']), dist)]
    
    while (len(dataset) < 10): # kind of an arbitrary cutoff point
        dist += 500
        dataset = dataset_recent.loc[bbox((latitude, longitude), (dataset_recent['latitude'], dataset_recent['longitude']), dist)]
    
    train, val = split_dataset(dataset)
    print('Training and validation sets created.')
    print(f'Size of training set: {len(train)}')
    print(f'Size of validation set: {len(val)}')

    prices = np.array(train['price'])

    # apply feature builder function to each row
    print('Creating feature vectors from training set...')
    features = np.array(train.apply(partial(build_feature, dist=dist, timedelta=365), axis=1).values.tolist())

    print('Performing linear regression...')
    model = sm.OLS(prices, features)
    results = model.fit()

    print(results.summary())
    logging.info(results.summary())

    preds = np.array(results.get_prediction(features).summary_frame(alpha=0.05)['mean'])
    r2 = r2_score(prices, preds)
    print(f'Training set: R2 = {r2}')
    logging.info(f'Training set: R2 = {r2}')

    actual_prices = np.array(val['price'])

    # iterate through the validation set, making prediction for each of them
    print('Creating feature vectors from validation set...')
    pred_features = np.array(val.apply(partial(build_feature, dist=dist, timedelta=365), axis=1).values.tolist())

    pred_prices = np.array(results.get_prediction(pred_features).summary_frame(alpha=0.05)['mean'])

    r2 = r2_score(actual_prices, pred_prices)
    print(f'Validation set: R2 = {r2}')
    logging.info(f'Validation set: R2 = {r2}')

    if r2 < 0.3:
        print('WARNING: low R2 value on validation set')
        logging.info('WARNING: low R2 value on validation set')

    test = {'latitude': latitude,
        'longitude': longitude,
        'date_of_transfer': date,
        'property_type': property_type}
    
    test_features = np.array([build_feature(test, dist=500, timedelta=365)])

    test_price = results.get_prediction(test_features).summary_frame(alpha=0.05)['mean'][0]
    return test_price
    pass


def predict_price(latitude, longitude, date, property_type):
    """
    Price prediction for UK housing.
    """

    # initialization
    result = _initialize()

    if result != 0:
        print('Initialization failed.')
        return

    print('Initialization successful.')

    # converting date to datetime format
    date = pd.to_datetime(date)
    
    # specify time range of data to retrieve
    start, end = max(1995, date.year - 2), min(2021, date.year + 3)

    global df

    # load relevant df
    if local:
        df_by_year = []
        for year in range(start, end):
            filepath = f'./data/pc-{year}.csv'
            df_by_year.append(assess.data(filepath=filepath))
        df = pd.concat(df_by_year, axis=0)
    else:
        df = assess.data(conn=conn, where=f"WHERE `date_of_transfer` >= '{start}-01-01' AND `date_of_transfer` <= '{end}-12-31'", local=False)

    # filter df based on location and property type
    df = df.loc[bbox((latitude, longitude), (df['latitude'], df['longitude']), 5000)]
    df = df.loc[df['property_type'] == property_type]

    # create dataset based on time and location
    dataset_recent = df.loc[(df['date_of_transfer'] - date).apply(lambda x: np.abs(x.days)) <= 365]

    # TODO: iteratively widen range if not enough data points?
    dist = 500
    dataset = dataset_recent.loc[bbox((latitude, longitude), (dataset_recent['latitude'], dataset_recent['longitude']), dist)]
    
    while (len(dataset) < 20): # kind of an arbitrary cutoff point
        dist += 500
        dataset = dataset_recent.loc[bbox((latitude, longitude), (dataset_recent['latitude'], dataset_recent['longitude']), dist)]

    prices = np.array(dataset['price'])

    # apply feature builder function to each row
    print('Creating feature vectors from training set...')
    features = np.array(dataset.apply(partial(build_feature, dist=dist, timedelta=365), axis=1).values.tolist())

    print('Performing linear regression...')
    model = sm.OLS(prices, features)
    results = model.fit()

    print(results.summary())
    logging.info(results.summary())

    preds = np.array(results.get_prediction(features).summary_frame(alpha=0.05)['mean'])
    r2 = r2_score(prices, preds)
    print(f'Training set: R2 = {r2}')
    logging.info(f'Training set: R2 = {r2}')

    test = {'latitude': latitude,
        'longitude': longitude,
        'date_of_transfer': date,
        'property_type': property_type}
    
    test_features = np.array([build_feature(test, dist=500, timedelta=365)])

    test_price = results.get_prediction(test_features).summary_frame(alpha=0.05)['mean'][0]
    return test_price
    pass
