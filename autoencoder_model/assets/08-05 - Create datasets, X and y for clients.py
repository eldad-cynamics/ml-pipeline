#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import gc
from datetime import datetime
from matplotlib import pyplot as plt

# from X_generator import *
from create_dataset_utils import *


# In[2]:


clients_devices = pd.read_csv('../clients_devices.csv')
clients_devices = clients_devices[clients_devices.status == 'active']
clients_devices


# In[17]:


bucket='rsrch-cynamics-datasets'

devices = [
#     ('Kcdda', 1, 0.004, 2),
     #('SickKids', 1, 0.01, 6),

    #('Guilford',1,0.01,1),
    #('Harrisonburg',1, 1/100,'6081cff532885f0009097bc3', '61f93abef66e8e0009967af5'),
    
    #('Coffee',1,1/100,4),
    
    #('Williamsburg', 1, 1/100, 3),
    #('Urbandale','FW',1/100,1),
    #('Saratoga','FW',1/100,1),
    #new walla devices 
    #('Walla', 'VTPocketASA', 1/100, 1),
    #('Walla', 'VTCLDUOASA', 1/100, 1)
    ('Walla', 1, 1/1000,1)

    
    #('Sioux', 1, 1/1000,1)
    #('JohnsonCity', 1, 1/1000,1)
    
#     ('Peachtree', 1, 0.01, 4),
#    ('Peachtree', 1, 0.01, 5),
#    ('Underline',1,0.03125,2)
    
    #('Rentpath','hq-fw', 1/100, 1)
    #('Rentpath','dev-fw', 1/100,1)
    
    #('Krypto','FW', 1/10, 1)

    
    #('HoustonCounty', 1, 1/50,2)
    #('MorganTown', 1, 1/10,1)
    
    #('Roseville', 'FW-Main', 1/1000,2),
    #('Roseville', 'FW-2', 1/1000,2)
        
    #('Christiansburg','FW',1/1000,1)
    #('Palmetto',1,1/100,1)        
    
]
client, device, sr, model_version = devices[0]
print(f'Working on {client} device={device} model={model_version}')


# In[14]:


days_by_type = {
'train': [],
   # '2022-02-17','2022-02-18','2022-02-19'],
'val': [],
    #'2022-02-20','2022-02-21'],
    #aviv: remove comment when adding test days
    'test':  ['2022-02-17',
 '2022-02-18',
 '2022-02-19',
 '2022-02-20',
 '2022-02-21',
 '2022-02-22',
 '2022-02-23',
 '2022-02-24',
 '2022-02-25',
 '2022-02-26',
 '2022-02-27',
 '2022-02-28',
 '2022-03-01',
 '2022-03-02',
 '2022-03-03',
 '2022-03-04',
 '2022-03-05',
 '2022-03-06',
 '2022-03-07',
 '2022-03-08',
 '2022-03-09',
 '2022-03-10',
 '2022-03-11',
 '2022-03-12',
 '2022-03-13',
 '2022-03-14']
}


# ## Read training, extract 'tracked_values'

# In[5]:


try:
    s3_filepath = f'clients/{client}/sr={sr}/device={device}/version=3/model={model_version}/type=train/tracked_values.json'
    tracked_values = eval(boto3.resource('s3').Object(bucket, s3_filepath).get()['Body'].read())
    print(f'Tracked values loaded successfully from {s3_filepath}')
except:
    training_df_list = {}
    for day in days_by_type['train']:
        print(f'Loading {day} from S3 ', end='...')
        sampled_df = pd.read_csv(f's3://{bucket}/clients/{client}/sr={sr}/device={device}/sampled/{day}.csv.gz')
        print('Done!', sampled_df.shape)

#         if day == '2020-07-27':
#             fp_start, fp_end = 1595856340, 1595882714
#             sampled_df = sampled_df[(sampled_df.time_window >= fp_start-60) & (sampled_df.time_window <= fp_end+60)]
        mean_packets_count = sampled_df.groupby('time_window').sum()['flowPackets'].mean()
        print(sampled_df.shape, mean_packets_count)
        training_df_list[day] = sampled_df

    print('Extracting tracked_values...', end='')
    tracked_values = get_tracked_values(pd.concat(training_df_list.values()), 
                                        f'clients/{client}/sr={sr}/device={device}/version=3/model={model_version}/type=train/',
                                        connections_tracking_size=10_000)
    print('Done!')
    del training_df_list


# ## Read each day, extract X & y for v1 and v2

# In[18]:


print(f'Working on {client} device={device}')
s3_bucket = boto3.resource('s3').Bucket('rsrch-cynamics-datasets')


# In[ ]:




for dataset_type, days in days_by_type.items():
#     if dataset_type != 'test':
#         continue
    print(f'----------------\nWorking on {dataset_type}\n----------------')
    for day in days:
        print(f'Loading {day} from S3 ', end='...')
        sampled_df = pd.read_csv(f's3://{bucket}/clients/{client}/sr={sr}/device={device}/sampled/{day}.csv.gz')
        print('Done!', sampled_df.shape)

        create_X_v1_v3(sampled_df, day, client, device, sr, model_version, dataset_type)


# In[ ]:




