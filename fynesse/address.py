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


def split_dataset(dataset, split=0.8):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    cutoff = int(split*len(dataset))
    train = dataset.iloc[indices[:cutoff]]
    val = dataset.iloc[indices[cutoff:]]

    return train, val
    pass


def retrieve_backing_set(df, row, dist, timedelta):
    """
    Retrieves the backing set of the specified row.
    """
    dataset = df.loc[assess.bbox((row['latitude'], row['longitude']), (df['latitude'], df['longitude']), dist)]
    dataset = dataset.loc[assess.recent(row['date_of_transfer'], dataset['date_of_transfer'], timedelta)]
    dataset = dataset.loc[row['property_type'] == dataset['property_type']]

    return dataset
    pass


def build_feature(df, row):
    backing_set = retrieve_backing_set(df, row, dist=500, timedelta=180)

    # timeseries and backing set features
    feature = [row['date_numeric'],
               np.mean(backing_set['price']),
               np.max(backing_set['price']),
               np.min(backing_set['price'])]
    
    # osmnx features
    """
    tags = {"amenity": True,
            "healthcare": True,
            "leisure": True,
            "shop": True}
    
    pois = access.retrieve_pois(latitude=row['latitude'], 
                                longitude=row['longitude'], 
                                tags=tags, 
                                dist=10000)
    
    if pois is None:
        for _ in range(5):
            feature.append(0)
    else:
        feature.append(len(pois))
        for key in tags.keys():
            if key in pois.columns:
                feature.append(len(pois[key].dropna()))
            else:
                feature.append(0)
    """

    tags = {"amenity": True}
    
    pois = access.retrieve_pois(latitude=row['latitude'], 
                                longitude=row['longitude'], 
                                tags=tags, 
                                dist=10000)
    
    if pois is None:
        feature.append(0)
    else:
        feature.append(len(pois))
        
    return feature
    pass


def predict_price(latitude, longitude, date, property_type, local=False, filepath='./data/prices-coordinates-data.csv', conn=None):
    """
    Price prediction for UK housing.
    """

    # converting input date to datetime format
    date = pd.to_datetime(date)

    if local:
        df = assess.data(local=local, filepath=filepath)
    else:
        df = assess.data(local=local, conn=conn)

    # filter df based on location and property type
    df = df.loc[assess.bbox((latitude, longitude), (df['latitude'], df['longitude']), 500)]
    df = df.loc[property_type == df['property_type']]

    # sort date column for later sampling
    df = df.sort_values(by='date_of_transfer')

    # create numeric date column
    date_numeric = assess.datetime_to_number(pd.concat([df['date_of_transfer'], pd.DataFrame([date])]))
    df['date_numeric'] = date_numeric.iloc[:-1]
    date_numeric = date_numeric.iloc[-1][0]

    # sample from entire dataset to build the training set - should this depend on the size of the dataset?
    if len(df) < 100:
        dataset = df
    else:
        indices = np.arange(0, len(df), step=len(df)//100)
        dataset = df.iloc[indices]

    print(f'Size of the training set: {len(dataset)}')
    logging.info(f'Size of the training set: {len(dataset)}')

    prices = np.array(dataset['price'])

    # apply feature builder function to each row
    print('Creating feature vectors from training set...')
    features = np.array(dataset.apply(partial(build_feature, df), axis=1).values.tolist())

    print('Performing linear regression...')
    model = sm.OLS(prices, features)
    results = model.fit()

    print(results.summary())
    logging.info(results.summary())

    preds = np.array(results.get_prediction(features).summary_frame(alpha=0.05)['mean'])
    r2 = r2_score(prices, preds)
    print(f'Training set: R2 = {r2:.3f}')
    logging.info(f'Training set: R2 = {r2:.3f}')

    test = {'latitude': latitude,
            'longitude': longitude,
            'date_of_transfer': date,
            'property_type': property_type,
            'date_numeric': date_numeric}
    
    features_test = np.array([build_feature(df, test)])

    preds_test = results.get_prediction(features_test).summary_frame(alpha=0.05)['mean'][0]
    return preds_test
    pass
