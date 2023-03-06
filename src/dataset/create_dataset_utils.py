import gc
import gzip
import os

import boto3
import pandas as pd
import psycopg2
from io import BytesIO, TextIOWrapper
from datetime import datetime
from sqlalchemy import create_engine
from matplotlib import pyplot as plt
from loguru import logger
from .X_generator import *


def redshift_query (query_order):
    """
    Connects to AWS Redshift, executes a query on a table, and returns the results as a Pandas dataframe.

    Args:
        redshift_endpoint (str): The endpoint URL of the Redshift cluster.
        redshift_port (str): The port number for the Redshift cluster.
        redshift_user (str): The username for the Redshift cluster.
        redshift_password (str): The password for the Redshift cluster.
        redshift_database (str): The name of the Redshift database.
        query_order (str): The SQL query order to be executed.

    Returns:
        A Pandas dataframe containing the results of the query, or None if an error occurs.
    """
    # Connect to Redshift

    redshift_database = os.getenv('redshift_database')
    redshift_host = os.getenv('redshift_host')
    redshift_port = os.getenv('redshift_port')
    redshift_user = os.getenv('redshift_user')
    redshift_password = os.getenv('redshift_password')

    try:
        conn = psycopg2.connect(
            host=redshift_host,
            port=redshift_port,
            user=redshift_user,
            password=redshift_password,
            database=redshift_database
        )
        logger.info('Connected to Redshift')
    except Exception as e:
        logger.exception('Error connecting to Redshift:', e)
        return None

    # Query a table
    try:
        df = pd.read_sql(query_order, conn)
        logger.info('Table query successful')
    except Exception as e:
        logger.exception('Error querying table:', e)
        df = None

    # Close the connection
    try:
        conn.close()
        logger.info('Connection closed')
    except Exception as e:
        logger.exception('Error closing connection:', e)

    return df


def save_csv_gz_to_s3 (df, s3_path, bucket='rsrch-cynamics-datasets'):
    logger.info(f'Saving to {s3_path} {df.shape}', end='... ', flush=True)

    buffer = BytesIO()
    with gzip.GzipFile(mode='w', fileobj=buffer) as zipped_file:
        df.to_csv(TextIOWrapper(zipped_file, 'utf8'), index=False)

    s3_object = boto3.resource('s3').Object(bucket, s3_path)
    s3_object.put(Body=buffer.getvalue())

    logger.info('Done writing csv_gz to s3!', flush=True)


def create_X_v1_v3 (sampled_df, day, client, device, sr, model_version, dataset_type,
                    time_window_size=5, sliding_window_size=20, bucket='rsrch-cynamics-datasets'):
    s3_bucket = boto3.resource('s3').Bucket(bucket)

    tracked_values_filepath = f'clients/{client}/sr={sr}/device={device}/version=3/model={model_version}/type=train/tracked_values.json'
    tracked_values = eval(boto3.resource('s3').Object(bucket, tracked_values_filepath).get()['Body'].read())

    mean_packet_count = sampled_df.groupby('time_window').sum()['flowPackets'].mean()
    print('Shape after sampling:', sampled_df.shape)
    print('Mean packet count:', mean_packet_count)

    datset_path = f'clients/{client}/device={device}/sr={sr}/type={dataset_type}/version=1'
    datset_path = f'clients/{client}/sr={sr}/device={device}/version=1/model={model_version}/type={dataset_type}/'
    print(f'Generating X1 for {day}', end='...')
    v1_features_df = extract_v1_features(sampled_df, time_window_size)
    X_v1 = generate_X(v1_features_df, sliding_window_size)
    print(f'Saving', end='...')
    y_v1 = pd.Series([np.nan] * X_v1.shape[0])

    save_dataset(X_v1, y_v1, datset_path, day)
    print(X_v1.shape, 'Done!')

    del X_v1
    gc.collect()

    datset_path = f'clients/{client}/sr={sr}/device={device}/version=3/model={model_version}/type={dataset_type}/'
    print(f'Generating X3 for {day}', end='...')
    v2_features_df = extract_v2_features(sampled_df, tracked_values, time_window_size)
    v3_features_df = pd.merge(v1_features_df, v2_features_df,
                              on='time_window', how='outer').fillna(0)

    del v1_features_df, v2_features_df
    gc.collect()

    X_v3 = generate_X(v3_features_df, sliding_window_size)
    print(f'Saving', end='...')
    y_v3 = pd.Series([np.nan] * X_v3.shape[0])
    save_dataset(X_v3, y_v3, datset_path, day)
    print(X_v3.shape, 'Done!')
    print(day, Counter(y_v1.fillna(False)))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title(f'{day}-device{device}', size=20)
    fig.set_size_inches(16, 9)
    y_v3.index = v3_features_df.index
    plot_attack = v3_features_df[y_v3.notnull()]
    ax.scatter(plot_attack.index, plot_attack['packetCount'], s=20, marker=(5, 2), c='red')
    ax.plot(v3_features_df.index, v3_features_df['packetCount'])
    plt.show()

    del v3_features_df, X_v3, y_v3

# def query_redshift (client, device, sr, day, account_id, device_id, time_window_size=5):
#     dbname = 'cynamics'
#     cluster = 'prodredshiftcluster.chk0levkctdj.us-east-1'
#     port = 5439
#     user = 'ro_user'
#     password = '37gvWDy5GZnwwdXjhGgQ'
#
#     connstr = f'postgresql://{user}:{password}@{cluster}.redshift.amazonaws.com:5439/cynamics'
#     engine = create_engine(connstr)
#
#     start = (datetime.strptime(day, '%Y-%m-%d') - datetime(1970, 1, 1)).total_seconds()
#     end = start + (24 * 60 * 60)
#
#     print(f'Querying {client}, device={device}, day={day}', end='...')
#     query = f"""
#     select creationtime, ipv4src, sourceport, ipprotocol, ipv4dest, destport, length, flowpackets
#     from public.rawdatas
#     where accountid='{account_id}'
#       and deviceid = '{device_id}'
#       and creationtime >= timestamp 'epoch' + {start} * interval '1 second'
#       and creationtime <  timestamp 'epoch' + {end}   * interval '1 second'
#       and samplingrate = {sr}
#     --limit 100"""
#
#     with engine.connect() as conn, conn.begin():
#         df = pd.read_sql(query, conn)
#     print('df shape: ', df.shape, end='...')
#
#     df['timestamp'] = (pd.to_datetime(df['creationtime']) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
#     df['time_window'] = df['timestamp'] - (df['timestamp'] % time_window_size)
#
#     print('Renaming', end='...')
#     df = df.rename(columns={'length': 'volume',
#                             'ipv4src': 'ipv4Src',
#                             'sourceport': 'sourcePort',
#                             'ipprotocol': 'ipProtocol',
#                             'ipv4dest': 'ipv4Dest',
#                             'destport': 'destPort',
#                             'flowpackets': 'flowPackets'
#                             })
#     df = df.groupby(['ipv4Src', 'sourcePort', 'ipProtocol',
#                      'ipv4Dest', 'destPort', 'time_window']) \
#         .agg({'flowPackets': sum,
#               'volume': sum
#               }).sort_values('time_window').reset_index()
#
#     save_csv_gz_to_s3(df, f'clients/{client}/sr={sr}/device={device}/sampled/{day}.csv.gz')
#
#     return df
