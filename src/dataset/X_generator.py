import os
import h5py
import copy
import boto3

import numpy as np
import pandas as pd

from functools import reduce
from collections import Counter

###################################
# features version 1
###################################
def bin_counter(flow_packets_column):
    # count flows by the bin of their length
    # 9 bins: 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, 129...
    return Counter(flow_packets_column.apply(lambda k: min(np.ceil(np.log2(k)) + 1, 9)))

def extract_NEWS_packets(df):
    split_point1 = df['time_window'].quantile(q=0.33)
    split_point2 = df['time_window'].quantile(q=0.66)

    first_third = df[df['time_window'] <= split_point1]
    first_third_df_NEW = pd.merge(first_third, first_third, 
                                  left_on=['time_window' ,'ipv4Src', 'ipv4Dest'],
                                  right_on=['prev_window' ,'ipv4Src', 'ipv4Dest'],
                                  how='left', indicator=True).query('_merge=="left_only"')

    second_third = df[(df['time_window'] >= split_point1) & (df['time_window'] <= split_point2)]
    second_third_df_NEW = pd.merge(second_third, second_third, 
                                   left_on=['time_window' ,'ipv4Src', 'ipv4Dest'],
                                   right_on=['prev_window' ,'ipv4Src', 'ipv4Dest'],
                                   how='left', indicator=True).query('_merge=="left_only"')

    last_third = df[df['time_window'] >= split_point2]
    last_third_df_NEW = pd.merge(last_third, last_third,
                                 left_on=['time_window' ,'ipv4Src', 'ipv4Dest'],
                                 right_on=['prev_window' ,'ipv4Src', 'ipv4Dest'],
                                 how='left', indicator=True).query('_merge=="left_only"')

    return pd.concat([first_third_df_NEW [first_third_df_NEW.time_window_x  != split_point1],
                      second_third_df_NEW[second_third_df_NEW.time_window_x != split_point2],
                      last_third_df_NEW])

def extract_v1_features(df, time_window_size):
    # Create 9 bins for flow_length
    bins_tmp_agg = df.groupby('time_window')\
                    .agg(flow_counter=pd.NamedAgg(column='flowPackets', aggfunc=bin_counter))\
                    .reset_index()
    bins = bins_tmp_agg.flow_counter.apply(pd.Series).fillna(0)
    for col in range(1, 10):
        if col not in bins.columns:
            bins[col] = 0.0
    bins.columns = [f'bin_{int(x)}' for x in bins.columns]
    bins = pd.concat([bins_tmp_agg['time_window'], bins], axis=1)

    # Create 9 bins for NEW flow_length
    df['prev_window'] = df['time_window'] - 5
    df_NEW = extract_NEWS_packets(df)
#     df_NEW = pd.merge(df, df, 
#                       left_on=['time_window' ,'ipv4Src', 'ipv4Dest'],
#                       right_on=['prev_window' ,'ipv4Src', 'ipv4Dest'],
#                       how='left', indicator=True).query('_merge=="left_only"')

    bins_NEWS_tmp_agg = df_NEW.groupby('time_window_x')\
                             .agg(flow_counter=pd.NamedAgg(column='flowPackets_x', aggfunc=bin_counter))\
                             .reset_index().rename(columns={'time_window_x':'time_window'})
    bins_NEWS = bins_NEWS_tmp_agg.flow_counter.apply(pd.Series).fillna(0)
    for col in range(1, 10):
        if col not in bins_NEWS.columns:
            bins_NEWS[col] = 0.0
    bins_NEWS.columns = [f'bin_NEWS_{int(x)}' for x in bins_NEWS.columns]
    bins_NEWS = pd.concat([bins_NEWS_tmp_agg['time_window'], bins_NEWS], axis=1)

    # merge bins to create features (18 bins)
    bins_df = pd.merge(bins, bins_NEWS, on='time_window', how='outer').fillna(0)

    # fill missing time windows
    bins_df = bins_df.set_index('time_window')\
                      .reindex(list(range(bins_df.time_window.min(), 
                                          bins_df.time_window.max()+1, 
                                          time_window_size)),
                               fill_value=0)

    return bins_df
###################################
# features version 2
###################################
def get_connections(df, ignore_time=False):
    tmp_df = df[['ipv4Src', 'ipv4Dest', 'flowPackets', 'time_window']].copy()
    tmp_df[['ipv4Src', 'ipv4Dest']] = np.sort(tmp_df[['ipv4Src', 'ipv4Dest']], axis=1)
    group_by_columns = ['ipv4Src', 'ipv4Dest'] if ignore_time else ['ipv4Src', 'ipv4Dest', 'time_window']
    retval = tmp_df.groupby(group_by_columns)\
                   .agg({'flowPackets':sum})\
                   .sort_values('flowPackets', ascending=False)\
                   .rename(columns={'flowPackets': 'totalPacketsInTraining'})
    return retval

def get_tracked_values(df, s3_filepath, connections_tracking_size, bucket='rsrch-cynamics-datasets'):
    security_ports = [1, 3, 7, 9, 13, 17, 19, 20, 21, 22, 23, 25, 26, 37, 42, 49, 53,
                     67, 68, 69, 79, 80, 81, 82, 88,
                     100, 106, 110, 111, 113, 119, 120, 123, 135, 136, 137, 138, 139,
                     143, 144, 158, 161, 162, 177, 179, 192, 194, 199, 254, 255, 280,
                     311, 389, 407, 427, 443, 444, 445, 464, 465, 497, 500, 513, 514,
                     515, 517, 518, 520, 543, 544, 548, 554, 587, 593, 623, 625, 626,
                     631, 636, 646, 664, 683, 787, 800, 808, 873, 902, 989, 990, 993,
                     995, 996, 997, 998, 999,
                     1000, 1001, 1008, 1019, 1021, 1022, 1023, 1024, 1025, 1026, 1027,
                     1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038,
                     1039, 1040, 1041, 1043, 1044, 1045, 1048, 1049, 1050, 1053, 1054,
                     1056, 1058, 1059, 1064, 1065, 1066, 1068, 1069, 1071, 1074, 1080,
                     1110, 1234, 1419, 1433, 1434, 1494, 1521, 1645, 1646, 1701, 1718,
                     1719, 1720, 1723, 1755, 1761, 1782, 1801, 1812, 1813, 1885, 1900,
                     1935, 1998, 2000, 2001, 2002, 2003, 2005, 2048, 2049, 2103, 2105,
                     2107, 2121, 2148, 2161, 2222, 2223, 2301, 2383, 2401, 2601, 2717,
                     2869, 2967, 3000, 3001, 3052, 3128, 3130, 3268, 3283, 3306, 3389,
                     3456, 3659, 3689, 3690, 3703, 3986, 4000, 4001, 4045, 4444, 4500,
                     4672, 4899, 5000, 5001, 5003, 5009, 5050, 5051, 5060, 5093, 5101,
                     5120, 5190, 5351, 5353, 5355, 5357, 5432, 5500, 5555, 5631, 5632,
                     5666, 5800, 5900, 5901, 6000, 6001, 6002, 6004, 6112, 6346, 6646,
                     6666, 7000, 7070, 7937, 7938, 8000, 8002, 8008, 8009, 8010, 8031,
                     8080, 8081, 8443, 8888, 9000, 9001, 9090, 9100, 9102, 9200, 9876,
                     9999]
    
    groupped = df.groupby('time_window')

    tracked_values = {
        'connections': get_connections(df, ignore_time=True).head(connections_tracking_size).to_dict(),
        'security_ports': security_ports,
        'protocols': list(range(50)),
        'median_packets_per_time_window': groupped.sum()['flowPackets'].median(),
        'median_volume_per_time_window' : groupped.sum()['volume'].median(),
        'median_flows_per_time_window'  : groupped.size().median(),
    }
    
    json_filename = 'tracked_values.json'
    with open(json_filename, 'w') as f:
        f.write(str(tracked_values))
    boto3.resource('s3').Bucket(bucket).upload_file(json_filename, 
                                                    os.path.join(s3_filepath, json_filename))
    os.remove(json_filename)
    
    # to read the information back to python, use:
    # eval(boto3.resource('s3').Object(bucket, s3_filepath).get()['Body'].read())
    
    return tracked_values

def aggregate_tracked_values(df, column_name, tracked_values):
    res_df = df.groupby(['time_window', column_name]).size().unstack(fill_value=0)
    for v in tracked_values:
        if v not in res_df:
            res_df[v] = 0
    res_df = res_df[tracked_values]
    res_df = res_df.div(res_df.sum(axis=1), axis=0).mul(100).round(3)
    res_df.columns = [f'{column_name}_{int(x)}' for x in res_df.columns]
    return res_df

def extract_v2_features(df, tracked_values, time_window_size):
    srcPort_df  = aggregate_tracked_values(df, 'sourcePort', tracked_values['security_ports'])
    destPort_df = aggregate_tracked_values(df, 'destPort',   tracked_values['security_ports'])
    protocol_df = aggregate_tracked_values(df, 'ipProtocol', tracked_values['protocols'])
    
    top_connections = pd.DataFrame.from_dict(tracked_values['connections'])
    top_connections.index.names = ['ipv4Src','ipv4Dest']
    connections_df = get_connections(df).reset_index().set_index(['ipv4Src', 'ipv4Dest'])

    new_connections_df = pd.merge(connections_df, top_connections,
                                  left_index=True, right_index=True,
                                  how='left', indicator=True
                                 ).query('_merge=="left_only"')\
                                  .groupby('time_window').size().to_frame('new_connections')
    
    statistics_df = df.groupby('time_window')\
                      .agg({'flowPackets': sum, 'ipv4Src': 'count', 'volume': sum})\
                      .rename(columns={'flowPackets': 'packetCount', 'ipv4Src': 'flowCount'})
    statistics_df['packetCount'] = statistics_df['packetCount'].div(tracked_values['median_packets_per_time_window'])
    statistics_df['flowCount'] = statistics_df['flowCount'].div(tracked_values['median_flows_per_time_window'])
    statistics_df['volume'] = statistics_df['volume'].div(tracked_values['median_volume_per_time_window'])
    
    v3_features_df = reduce(lambda left,right: pd.merge(left , right, on='time_window', how='outer'), 
                            [statistics_df, new_connections_df,
                             srcPort_df, destPort_df, protocol_df]).fillna(0)
    
    # fill missing time windows
    v3_features_df = v3_features_df.reindex(list(range(v3_features_df.index.min(), 
                                                       v3_features_df.index.max()+1, 
                                                       time_window_size)),
                                            fill_value=0)

    return v3_features_df

###################################
# Generate X and save
###################################

def generate_X(features_df, sliding_window_size):
    df = features_df.reset_index(drop=True)
    
    # pad intro with zeros
    for i in range(-sliding_window_size,0):
        df.loc[i] = 0
    df.sort_index(inplace=True)

    # generate X
    X = np.stack([df.loc[i-sliding_window_size+1:i,:] for i in range(df.index.max()+1)])
    
    return X

def save_dataset(X, y, s3_filepath, filename, bucket='rsrch-cynamics-datasets'):
    h5_filename = f'{filename}.h5'
    with h5py.File(h5_filename, 'w') as h5f:
        h5f.create_dataset('X', data=X)
    boto3.resource('s3').Bucket(bucket).upload_file(h5_filename, 
                                                    os.path.join(s3_filepath, h5_filename)
                                                   )
    os.remove(h5_filename)

    # save labels
    csv_filename = f'{filename}.csv'
    y.to_csv(os.path.join('s3://', bucket, s3_filepath, csv_filename), index=False, header=['label'])

def get_attack_app(timestamp, attacks_df):
    s1 = timestamp
    e1 = timestamp + 10
    tuples = zip(attacks_df['Attack Name'], 
                 attacks_df['start_timestamp'], 
                 attacks_df['end_timestamp'])
    for app, s2, e2 in tuples:
        if  (s2 <= s1 and e2 >= s1) or \
            (s2 <= e1 and e2 >= e1) or \
            (s2 >= s1 and e2 <= e1):
                return app
    return np.nan
