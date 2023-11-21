import numpy as np
import pandas as pd

from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


# store table fields as lists
pp_fields = ['transaction_unique_identifier', 'price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'primary_addressable_object_name', 'secondary_addressable_object_name', 'street', 'locality', 'town_city', 'district', 'county', 'ppd_category_type', 'record_status']
postcode_fields = ['postcode', 'status', 'usertype', 'easting', 'northing', 'positional_quality_indicator', 'country', 'latitude', 'longitude', 'postcode_no_space','postcode_fixed_width_seven', 'postcode_fixed_width_eight', 'postcode_area', 'postcode_district', 'postcode_sector', 'outcode', 'incode']
pc_fields = ['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude']


def dist(point1, point2, dist):
    """
    Returns whether point2 is within some distance N, S, E, W of point1.
    :param point1: a (lat, lon) tuple
    :param point2: a (lat, lon) tuple
    :param dist: distance specified in meters
    """
    # convert meters to degrees
    dist = dist / 1000 / (40075/360)
    return np.abs(point2[0]-point1[0]) <= dist and np.abs(point2[1]-point1[1]) <= dist
    pass


def data(conn=None, table='prices_coordinates_data', filepath='./data/prices-coordinates-data.csv', local=True):
    """
    Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame.
    """
    df = access.data(conn, table, filepath, local)
    df.columns = pc_fields

    df['date_of_transfer']= pd.to_datetime(df['date_of_transfer'])
    df['property_type'] = df['property_type'].astype('category')
    df['new_build_flag'] = df['new_build_flag'].astype('category')
    df['tenure_type'] = df['tenure_type'].astype('category')
    return df
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
