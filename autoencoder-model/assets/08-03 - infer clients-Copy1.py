#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from inference_utils import *


# In[2]:


clients_devices = [
     #('Kcdda',1,0.004,2),
#     ('SickKids',1,0.01,1),
     #('SickKids',1,0.01,6),
     ('Guilford',1,0.01,1)
    #('Harrisonburg',1, 1/1000,2)
    
    #('Sioux',1,1/1000,1)
    #('JohnsonCity', 1, 1/1000,1)
    #('Walla', 1, 1/1000,1)
    
#     ('Peachtree',1,0.01,5),
    
    #('Roseville', 'FW-Main', 1/1000,2),
    #('Roseville', 'FW-2', 1/1000,2)
    
    #('HoustonCounty', 1, 1/50,2)
    #('Krypto', 'FW', 1/10,1)
    #('MorganTown', 1, 1/10,1)

    #('Coffee',1,1/100,4)
    #('Christiansburg','FW',1/1000,1)
    #('Palmetto',1,1/100,1)
    
    #('Williamsburg', 1, 1/100, 3),
    #('Urbandale','FW',1/100,1),
    #('Saratoga','FW',1/100,1),
    #('Walla', 'VTPocketASA', 1/100,1),
    #('Walla', 'VTCLDUOASA', 1/100, 1)

]


# In[3]:


client, device, sr, model_version = clients_devices[0]
get_device_details(client, device, sr, model_version)


# In[4]:


newdays=['2022-02-17',
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
#newdays=None


# In[ ]:


for client, device, sr, model_version in clients_devices:
    model_version, _, _, v1_model_name, v3_model_name, _ = get_device_details(client, device, sr, model_version)
    #comment the training job in utils py file if not from sage
    infer_using_tuned_autoencoder(client, device, sr, 1, model_version,
                                  v1_model_name, days=[])
    
    infer_using_tuned_autoencoder(client, device, sr, 3, model_version,
                                  v3_model_name, days=[])

    print('--------------------')


# In[ ]:




