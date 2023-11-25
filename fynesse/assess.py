import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes sure they are correctly labeled. How is the data indexed. Create visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


# store table fields as lists
pp_fields = ['transaction_unique_identifier', 'price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'primary_addressable_object_name', 'secondary_addressable_object_name', 'street', 'locality', 'town_city', 'district', 'county', 'ppd_category_type', 'record_status']
postcode_fields = ['postcode', 'status', 'usertype', 'easting', 'northing', 'positional_quality_indicator', 'country', 'latitude', 'longitude', 'postcode_no_space','postcode_fixed_width_seven', 'postcode_fixed_width_eight', 'postcode_area', 'postcode_district', 'postcode_sector', 'outcode', 'incode']
pc_fields = ['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude']


def data(conn=None, table='prices_coordinates_data', where=None, filepath='./data/prices-coordinates-data.csv', local=True):
    """
    Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame.
    """
    df = access.data(conn, table, where, filepath, local)
    df.columns = pc_fields

    df['date_of_transfer']= pd.to_datetime(df['date_of_transfer'])
    df['property_type'] = df['property_type'].astype('category')
    df['new_build_flag'] = df['new_build_flag'].astype('category')
    df['tenure_type'] = df['tenure_type'].astype('category')
    return df
    pass

def sample_from(df, n):
    """
    Returns a list of indices.
    """
    indices = np.arange(len(df))
    samples = np.random.choice(indices, size=n, replace=False)
    return samples
    pass

def datetime_to_number(column):
    """
    Transform a column of datetime type to number type.
    """
    return (column - column.min()) / np.timedelta64(1,'D')
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


def recent(date1, date2, timedelta):
    """
    Returns whether date2 is within some timedelta away from date1.
    :param date1: a datetime value
    :param date2: a datetime value
    :param timedelta: specified in days
    """
    return (date2-date1).apply(lambda x: np.abs(x.days)) <= timedelta
    pass


def plot_line(ax, slope, intercept, start, stop, color):
    x = np.array([start, stop])
    try:
        y = slope * x + intercept
    except:
        y = slope * datetime_to_number(x) + intercept
    ax.plot(x, y, color=color)
    pass


def plot_distribution(ax, data, bins=50):
    """
    Graphs the distribution of the data, which is assumed to be one-dimensional.
    """

    mean = np.mean(data)
    var = np.var(data)

    ax.hist(data, bins=bins)
    ax.axvline(mean, color='orange', linestyle='dashed')
    ax.axvline(np.quantile(data, .25), color='orange', linestyle='dashed', alpha=0.5)
    ax.axvline(np.quantile(data, .50), color='orange', linestyle='dashed', alpha=0.75)
    ax.axvline(np.quantile(data, .75), color='orange', linestyle='dashed', alpha=0.5)

    ax.set_title(f'mean = {mean:.3f}, variance = {var:.3f}')

    return mean, var

    pass


def plot_correlation(ax, target, feature, regression=False):

    ax.scatter(feature, target)

    if regression:
        try:
            basis = np.reshape(feature, (-1, 1))
            basis = sm.add_constant(basis)
        except: # probably a datetime type, need to cast to numeric
            basis = np.reshape(datetime_to_number(feature), (-1, 1))
            basis = sm.add_constant(basis)

        model = sm.OLS(target, basis)
        results = model.fit()

        intercept = results.params['const']
        slope = results.params['x1']
        r2 = results.rsquared

        plot_line(ax, slope, intercept, np.min(feature), np.max(feature), color='orange')
        ax.set_title(f'R2 = {r2:.3f}')

        return r2, (slope, intercept)

    pass

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
