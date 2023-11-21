import urllib.request
import pymysql
import os
from tabulate import tabulate
import pandas as pd
import osmnx as ox

from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """


def create_connection(user, password, host, database, port=3306):
    """
    Create a database connection to the MariaDB database specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")

    return conn
    pass


def execute_query(conn, query, multi_line=False, fetch_rows=True):
    """
    :param conn: handle to database connection
    :param queries: a string representing a single-line query, or a tuple of strings representing a multi-line query
    :param fetch_rows: do you want any rows to be returned?
    """
    cursor = conn.cursor()

    if multi_line:
        count = cursor.execute('\n'.join(query))
    else:
        count = cursor.execute(query)

    print(f'{count} rows affected.')

    if fetch_rows:
        result = cursor.fetchall()
        print(tabulate(result[:5]))
        return result

    pass


def download_pp_data(path='.'):
    """
    Retrieves the csvs containing the the price paid data and stores them under the supplied path.
    """
    for year in range(1995, 2022):
        for part in range(1, 3):
            filename = f'pp-{year}-part{part}.csv'
            filepath = os.path.join(path, filename)
            urllib.request.urlretrieve(f'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/{filename}', filename=filepath)
    pass


def download_postcode_data(path='.'):
    filepath = os.path.join(path, 'open_postcode_geo.csv.zip')
    urllib.request.urlretrieve('https://www.getthedata.com/downloads/open_postcode_geo.csv.zip', filename=filepath)
    os.system(f'unzip {filepath} -d {path}')
    pass


def upload_data(conn, table, filepath, clean=False):
    """
    :param clean: boolean value denoting whether the specified csv file is *clean*!
    Uploads the specified csv file to the specified table. Note that the SQL query is not the most efficient version for general well-behaving csv files; 
    """
    if clean:
        query = (f"LOAD DATA LOCAL INFILE '{filepath}' INTO TABLE `{table}`",
                 "FIELDS TERMINATED BY ','",
                 "LINES STARTING BY '' TERMINATED BY '\n';")
    else:
        query = (f"LOAD DATA LOCAL INFILE '{filepath}' INTO TABLE `{table}`",
                 "FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"'",
                 "LINES STARTING BY '' TERMINATED BY '\n';")
        
    execute_query(conn, query=query, multi_line=True, fetch_rows=False)
    pass


def initialize_database(conn):
    queries = ["SET SQL_MODE = \"NO_AUTO_VALUE_ON_ZERO\";",
               "SET time_zone = \"+00:00\";",
               "CREATE DATABASE IF NOT EXISTS `property_prices` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;",
               "USE `property_prices`;"]
    
    for query in queries:
        execute_query(conn, query=query, multi_line=False, fetch_rows=False)

    print('Database initialized successfully.')
    pass


def initialize_table(conn, table):
    """
    Initializes the table schema and sets up the primary key.
    """

    match table:
        case 'pp_data':
            queries = ["USE `property_prices`;",
                       "DROP TABLE IF EXISTS `pp_data`;",
                       ("CREATE TABLE IF NOT EXISTS `pp_data` (",
                        "`transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,",
                        "`price` int(10) unsigned NOT NULL,",
                        "`date_of_transfer` date NOT NULL,",
                        "`postcode` varchar(8) COLLATE utf8_bin NOT NULL,",
                        "`property_type` varchar(1) COLLATE utf8_bin NOT NULL,",
                        "`new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,",
                        "`tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,",
                        "`primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,",
                        "`secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,",
                        "`street` tinytext COLLATE utf8_bin NOT NULL,",
                        "`locality` tinytext COLLATE utf8_bin NOT NULL,",
                        "`town_city` tinytext COLLATE utf8_bin NOT NULL,",
                        "`district` tinytext COLLATE utf8_bin NOT NULL,",
                        "`county` tinytext COLLATE utf8_bin NOT NULL,",
                        "`ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,",
                        "`record_status` varchar(2) COLLATE utf8_bin NOT NULL,",
                        "`db_id` bigint(20) unsigned NOT NULL",
                        ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1;"),
                       ("ALTER TABLE `pp_data`",
                        "ADD PRIMARY KEY (`db_id`);"),
                       ("ALTER TABLE `pp_data`",
                        "MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1;")]

        case 'postcode_data':
            queries = ["USE `property_prices`;",
                       "DROP TABLE IF EXISTS `postcode_data`;",
                       ("CREATE TABLE IF NOT EXISTS `postcode_data` (",
                        "`postcode` varchar(8) COLLATE utf8_bin NOT NULL,",
                        "`status` enum('live','terminated') NOT NULL,",
                        "`usertype` enum('small', 'large') NOT NULL,",
                        "`easting` int unsigned,",
                        "`northing` int unsigned,",
                        "`positional_quality_indicator` int NOT NULL,",
                        "`country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,",
                        "`latitude` decimal(11,8) NOT NULL,",
                        "`longitude` decimal(10,8) NOT NULL,",
                        "`postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,",
                        "`postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,",
                        "`postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,",
                        "`postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,",
                        "`postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,",
                        "`postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,",
                        "`outcode` varchar(4) COLLATE utf8_bin NOT NULL,",
                        "`incode` varchar(3)  COLLATE utf8_bin NOT NULL,",
                        "`db_id` bigint(20) unsigned NOT NULL",
                        ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin;"),
                       ("ALTER TABLE `postcode_data`",
                        "ADD PRIMARY KEY (`db_id`);"),
                       ("ALTER TABLE `postcode_data`",
                        "MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;")]

        case 'prices_coordinates_data':
            queries = ["USE `property_prices`;",
                       "DROP TABLE IF EXISTS `prices_coordinates_data`;",
                       ("CREATE TABLE IF NOT EXISTS `prices_coordinates_data` (",
                        "`price` int(10) unsigned NOT NULL,",
                        "`date_of_transfer` date NOT NULL,",
                        "`postcode` varchar(8) COLLATE utf8_bin NOT NULL,",
                        "`property_type` varchar(1) COLLATE utf8_bin NOT NULL,",
                        "`new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,",
                        "`tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,",
                        "`locality` tinytext COLLATE utf8_bin NOT NULL,",
                        "`town_city` tinytext COLLATE utf8_bin NOT NULL,",
                        "`district` tinytext COLLATE utf8_bin NOT NULL,",
                        "`county` tinytext COLLATE utf8_bin NOT NULL,",
                        "`country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,",
                        "`latitude` decimal(11,8) NOT NULL,",
                        "`longitude` decimal(10,8) NOT NULL,",
                        "`db_id` bigint(20) unsigned NOT NULL",
                        ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1;")]

        case _:
            raise ValueError

    for (i, query) in enumerate(queries):
        execute_query(conn, query=query, multi_line=(i>=2), fetch_rows=False)
    pass


def create_index(conn, table, index_name, columns):
    """
    Create index with name index_name on the specified table from the specified columns.
    :param conn:
    :param table:
    :param index_name:
    :param columns: a list of strings
    """
    query = (f"CREATE INDEX {index_name}",
             f"ON `{table}` ({'`' + '`, `'.join(columns) + '`'});")
    execute_query(conn, query=query, multi_line=True, fetch_rows=False)
    pass


def retrieve_pois(place_name, latitude, longitude, tags, dist):
    """
    Create GeoDataFrame of OSM features within some distance N, S, E, W of a point.
    :param center_point: the (lat, lon) center point around which to get the features
    :param tags: Dict of tags used for finding elements in the selected area. Results returned are the union, not intersection of each individual tag. Each result matches at least one given tag. The dict keys should be OSM tags, (e.g., building, landuse, highway, etc) and the dict values should be either True to retrieve all items with the given tag, or a string to get a single tag-value combination, or a list of strings to get multiple values for the given tag. 
    :param dist: distance in meters
    """

    pois = ox.features_from_point((latitude, longitude), tags, dist)
    print(f"There are {len(pois)} points of interest surrounding {place_name} latitude: {latitude}, longitude: {longitude}")
    return pois
    pass


def data(conn=None, table='prices_coordinates_data', filepath='./data/prices-coordinates-data.csv', local=True):
    """
    Read the data from the web or local file, returning structured format as a dataframe
    :param conn:
    :param table:
    :param path: path to the directory containing the data files
    :param local: a boolean specifying whether the data should be read from a local file
    """
    if local:
        df = pd.read_csv(filepath)

    else:
        query = f"SELECT * FROM `{table}`;"
        df = pd.read_sql(query, conn)

    return df
    pass

