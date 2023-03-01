#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import boto3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# In[2]:


def list_loss_paths(client, device, sampling_rate, features_version, tuned_model_name, bucket):
    #device_path = f'clients/{client}/device={device}/sr={sampling_rate}/'
    device_path = f'clients/{client}/sr={sampling_rate}/device={device}'
    
    retval = {}
    s3_bucket = boto3.resource('s3').Bucket(bucket)
    all_device_files = list(s3_bucket.objects.filter(Prefix=device_path))

    all_loss_files = [x.key for x in all_device_files 
                      if tuned_model_name in x.key
                      and '.loss' in x.key
                     ]
    
    for loss_file_path in all_loss_files:
        filename = loss_file_path.split('/')[-1].split('.')[0]
        try:
            label_file_path = [x.key for x in all_device_files 
                               if f'{filename}.csv' in x.key 
                               and f'version={features_version}' in x.key][0]
        except:
            continue

        retval[filename] = (loss_file_path, label_file_path)
    
    return retval


# In[3]:


bucket = 'rsrch-cynamics-datasets'

devices_and_models = [
    #('Fayette',   182,   0.001,     '06-24-Fayette-182-v1-027-79b0045e', '06-24-Fayette-182-v3-033-faf8fb1d'),
    #('Fayette',   228,   0.001,     '06-24-Fayette-228-v1-012-57d1205a', '06-24-Fayette-228-v3-002-b0d80e4c'),
    
    #('Kcdda',     1,     0.004,     '06-24-Kcdda-1-v1-048-f9076f00',     '06-24-Kcdda-1-v3-001-ec8de1a5'),
    #('Kcdda',     1,     0.004, '04-12-Kcdda-1-v1-002-830544d0','04-12-Kcdda-1-v3-009-4162de2f'),
    
    #('Peachtree', 1,     0.01,      '09-16-Peachtree-1-v1-004-55d9e3c8', '09-16-Peachtree-1-v3-032-a01d1236'),
    #('Underline',1,0.03125,'09-23-aviv-under-Underline-1-v1-017-422c111f','09-23-aviv-under-Underline-1-v3-012-ee30434a'),
    
    #('SickKids',  1,     0.01,  '01-29-sk-SickKids-1-v1-038-6ae85743','01-29-sk-SickKids-1-v3-026-f8b8b245'),
    #('SickKids',  1,     0.01,  '06-13-SickKids-1-v1-016-a6e6643f','06-13-SickKids-1-v3-019-7b79f409'),
    
    #('Harrisonburg',1,1/1000,'10-03-Harrisonburg-1-v1-010-1ccaabef','10-03-Harrisonburg-1-v3-022-cc67c44a'),

    
    #('JohnsonCity',1,1/1000,'06-21-JohnsonCity-1-v1-014-fa685b46','06-21-JohnsonCity-1-v3-002-14de8a60'),    
    #('Sioux',1,1/1000,'06-22-Sioux-1-v1-029-9d0b5553','06-22-Sioux-1-v3-010-4616c69a'),
    #('Walla',1,1/1000,'06-16-Walla-1-v1-006-fc264554','06-16-Walla-1-v3-007-d10964ba')
    
    ('Guilford',1,0.01,'03-16-Guilford-1-v1-003-336da5d6','03-16-Guilford-1-v3-007-5ab0ca39')
    
    #('Roseville','FW-Main',1/1000,'06-24-Roseville-FW-Main-v1-028-65a05e34','06-24-Roseville-FW-Main-v3-013-adf1a6ed'),
    #('Roseville','FW-2',1/1000,'06-24-fw2-Roseville-FW-2-v1-032-75bc498e','06-24-fw2-Roseville-FW-2-v3-034-5523cf38')
    
    #('Coffee',1,0.01,'10-03-Coffee-1-v1-001-56768a1f','10-03-Coffee-1-v3-006-a836e8bb')
    
    #('HoustonCounty',1,0.02,'07-27-HoustonCounty-1-v1-036-e10753d0','07-27-HoustonCounty-1-v3-016-0e9d776f'),
    #('MorganTown',1,0.1,'04-26-M2-MorganTown-1-v1-018-2fd0a21e','03-09-MORG-MorganTown-1-v3-034-91e0c72e')
   #('Christiansburg','FW',0.001,'04-11-Christiansburg-FW-v1-013-8af25341','03-30-Clarks-Clarksville-FW-v3-014-015ee306')    
    #('Palmetto',1,1/100,'05-20-Palmetto-1-v1-029-4f48d948','05-20-Palmetto-1-v3-011-131a28a4')
    
    
    #('Williamsburg',1,1/100,'12-26-Williamsburg-1-v1-011-4f5a4f4f','12-26-Williamsburg-1-v3-025-84ee8c2b')
    #('Urbandale','FW',1/100,'12-26-Urbandale-FW-v1-011-e23e71d8','12-26-Urbandale-FW-v3-008-d3e54ac1'),
    #('Saratoga','FW',1/100,'12-26-Saratoga-FW-v1-007-94a5e215','12-26-Saratoga-FW-v3-006-680275e3'),
    #new walla devices 
    #('Walla', 'VTPocketASA', 1/100, '12-26-Walla-VTPocketASA-v1-010-ad4072d6','12-26-Walla-VTPocketASA-v3-009-251d720c'),
    #('Walla', 'VTCLDUOASA', 1/100, '12-26-Walla-VTCLDUOASA-v1-026-29f2d7bd','12-26-Walla-VTCLDUOASA-v3-030-1d2f638c')


]


# In[4]:


colors = list(mcolors.TABLEAU_COLORS.values())

client, device, sampling_rate, tuned_v1_model_name, tuned_v3_model_name = devices_and_models[-1]
print(f'Working on {client}, {device}')

loss_paths_dictionary = list_loss_paths(client, device, sampling_rate, 1, tuned_v1_model_name, bucket)


# In[5]:


tuned_v1_model_name


# In[6]:


for day, v in sorted(loss_paths_dictionary.items()):
    loss_file_path, label_file_path = v

    # get loss file
    boto3.client('s3').download_file(bucket, loss_file_path, 'temp_file.loss')
    with h5py.File('temp_file.loss', 'r') as h5f:
        loss = h5f['loss'][:]

    # get labels
    y = pd.read_csv(f's3://{bucket}/{label_file_path}')['label'].fillna(False)
    if label_file_path=='clients/Coffee/sr=0.01/device=1/version=1/model=2/type=test/2021-05-23.csv':
        continue
    print(loss.shape, y.shape)
    
    # plot
    to_plot = pd.DataFrame({'loss':loss.mean(axis=1), 'attack':y})
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title(f'{client}-{device} {day}', size=20)
    fig.set_size_inches(16,9)
    ax.plot(to_plot.index, to_plot.loss)
    for i, attack in enumerate(to_plot.attack.unique()):
        if attack != False:
            plot_attack = to_plot[to_plot.attack == attack]
            ax.scatter(plot_attack.index, plot_attack.loss, s=20, marker=(5, 2), c=colors[i%len(colors)], label=attack)
    fig.legend()
    plt.show()


# In[7]:


loss_paths_dictionary


# In[8]:


for day, v in sorted(loss_paths_dictionary.items()):
    loss_file_path, label_file_path = v
    print(label_file_path)


# In[36]:


y


# In[ ]:





# In[ ]:





# In[ ]:




