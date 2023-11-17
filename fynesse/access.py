import urllib.request
import pymysql
import os
from tabulate import tabulate

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


def execute_queries(conn, queries, fetch_rows=False):
    """
    :param conn: handle to database connection
    :param queries: a list of tuples, with each tuple potentially representing a multi-line query
    :param fetch_rows: do you want any rows to be returned?
    """

    for query in queries:
        cursor = conn.cursor()
        if fetch_rows:
            cursor.execute('\n'.join(query))
            print(tabulate(cursor.fetchall()))
        else:
            count = cursor.execute('\n'.join(query))
            print(f'{count} rows affected.')
    pass


def download_pp_data(path):
    """
    Retrieves the csvs containing the the price paid data and stores them under the supplied path.
    """
    if path is None:
        path = '.'
    
    for year in range(1995, 2022):
        for part in range(1, 3):
            filename = f'pp-{year}-part{part}.csv'
            filepath = os.path.join(path, filename)
            urllib.request.urlretrieve(f'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/{filename}', filename=filepath)
    pass


def upload_pp_data(conn, path):
    """
    Uploads the csvs containing the price paid data to the AWS database.
    """
    if path is None:
        path = '.'

    for year in range(1995, 2022):
        print(f'Uploading data from year {year}...')
        for part in range(1, 3):
            cursor = conn.cursor()
            filepath = os.path.join(path, f'pp-{year}-part{part}.csv')
            query = (f"LOAD DATA LOCAL INFILE '{filepath}' INTO TABLE `pp_data`",
                     "FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"'",
                     "LINES STARTING BY '' TERMINATED BY '\n';")
            count = cursor.execute('\n'.join(query))
            print(f'{count} rows affected.')
    pass


def download_postcode_data(path):
    if path is None:
        path = '.'
    filepath = os.path.join(path, 'open_postcode_geo.csv.zip')
    urllib.request.urlretrieve('https://www.getthedata.com/downloads/open_postcode_geo.csv.zip', filename=filepath)
    os.system(f'unzip {filepath} -d {path}')
    pass


def upload_postcode_data(conn, path):
    if path is None:
        path = '.'

    cursor = conn.cursor()
    filepath = os.path.join(path, 'open_postcode_geo.csv')
    query = (f"LOAD DATA LOCAL INFILE '{filepath}' INTO TABLE `postcode_data`",
             "FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"'",
             "LINES STARTING BY '' TERMINATED BY '\n';")
    count = cursor.execute('\n'.join(query))
    print(f'{count} rows affected.')
    pass


def initialize_database(conn):
    queries = ("SET SQL_MODE = \"NO_AUTO_VALUE_ON_ZERO\";",
             "SET time_zone = \"+00:00\";",
             "CREATE DATABASE IF NOT EXISTS `property_prices` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;",
             "USE `property_prices`;")
    
    for query in queries:
        cursor = conn.cursor()
        count = cursor.execute(query)
        print(f'{count} rows affected.')
    pass


def initialize_table(conn, table):
    """
    Initializes the table schema and sets up the primary key.
    """
    match table:
        case 'pp_data':
            # setting up the schema for table `pp_data`
            count = conn.cursor().execute("USE `property_prices`;")
            print(f'{count} rows affected.')

            count = conn.cursor().execute("DROP TABLE IF EXISTS `pp_data`;")
            print(f'{count} rows affected.')

            cursor = conn.cursor()
            query = ("CREATE TABLE IF NOT EXISTS `pp_data` (",
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
                     ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1;")
            count = cursor.execute('\n'.join(query))
            print(f'{count} rows affected.')

            # adding the primary key for table `pp_data`
            cursor = conn.cursor()
            query = ("ALTER TABLE `pp_data`",
                     "ADD PRIMARY KEY (`db_id`);")
            count = cursor.execute('\n'.join(query))
            print(f'{count} rows affected.')

            cursor = conn.cursor()
            query = ("ALTER TABLE `pp_data`",
                     "MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1;")
            count = cursor.execute('\n'.join(query))
            print(f'{count} rows affected.')

        case 'postcode_data':
            # setting up the schema for table `postcode_data`
            count = conn.cursor().execute("USE `property_prices`;")
            print(f'{count} rows affected.')

            count = conn.cursor().execute("DROP TABLE IF EXISTS `postcode_data`;")
            print(f'{count} rows affected.')
            
            cursor = conn.cursor()
            query = ("CREATE TABLE IF NOT EXISTS `postcode_data` (",
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
                     ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin;")
            count = cursor.execute('\n'.join(query))
            print(f'{count} rows affected.')

            # adding the primary key
            cursor = conn.cursor()
            query = ("ALTER TABLE `postcode_data`",
                     "ADD PRIMARY KEY (`db_id`);")
            count = cursor.execute('\n'.join(query))
            print(f'{count} rows affected.')

            cursor = conn.cursor()
            query = ("ALTER TABLE `postcode_data`",
                     "MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;")
            count = cursor.execute('\n'.join(query))
            print(f'{count} rows affected.')

        case _:
            raise ValueError

    pass


def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

