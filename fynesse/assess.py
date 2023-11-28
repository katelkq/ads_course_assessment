import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
import osmnx as ox
import folium
from folium.plugins import HeatMap

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
    return (column - np.min(column)) / np.timedelta64(1,'D')
    pass


def meter_to_degree(x):
    return x / 1000 / (40075/360)
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


def plot_line(ax, slope, intercept, x, color):
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
    ax.axvline(np.quantile(data, .25), color='pink', linestyle='dashed', alpha=0.75)
    ax.axvline(np.quantile(data, .50), color='pink', linestyle='dashed')
    ax.axvline(np.quantile(data, .75), color='pink', linestyle='dashed', alpha=0.75)

    ax.set_title(f'mean = {mean:.3f}, variance = {var:.3f}')

    return mean, var

    pass


def plot_correlation(ax, target, feature, label=None, regression=False):

    ax.scatter(feature, target, label=label)

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

        plot_line(ax, slope, intercept, feature, color='orange')
        ax.set_title(f'R2 = {r2:.3f}')

        return r2, (slope, intercept)

    pass


def plot_geography(ax, place_name, latitude, longitude, pois, dist):
    graph = ox.graph_from_point((latitude, longitude), dist=dist)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    # Get place boundary related to the place name as a geodataframe
    area = ox.geocode_to_gdf(place_name)

    # Plot the footprint
    area.plot(ax=ax, facecolor="white")

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    radius = meter_to_degree(10000)
    ax.set_xlim([longitude-radius, longitude+radius])
    ax.set_ylim([latitude-radius, latitude+radius])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    ax.scatter(longitude, latitude, color='m')

    # Plot all POIs
    pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    pass


def plot_geo_heatmap(df, target):
    # Load your data into a pandas DataFrame
    # Assuming your DataFrame is named df with columns: 'latitude', 'longitude', 'price'
    # Replace this with your actual data loading process

    # Create a map centered around the UK
    uk_map = folium.Map(location=[54.7023545, -3.2765753], zoom_start=6)

    # Ensure the data is aggregated by location (latitude, longitude) and house prices
    heat_data = df[['latitude', 'longitude', target]].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist()

    # Plot heatmap using HeatMap from folium.plugins
    HeatMap(heat_data, radius=10, blur=15).add_to(uk_map)

    # Save the map as an HTML file or display it
    # uk_map.save("prices_heatmap.html")
    return uk_map
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
