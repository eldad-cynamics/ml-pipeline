#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
from create_dataset_utils import query_redshift, create_X_v1_v3


# In[3]:


clients_devices = pd.read_csv('../clients_devices.csv')
clients_devices = clients_devices[clients_devices.status == 'active']
clients_devices


# In[4]:


from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield (start_date + timedelta(n)).strftime("%Y-%m-%d")

days = list(daterange(date(2022, 2, 13), date(2022, 3, 15)))
#days = list(daterange(date(2021,10,6), date(2021, 10, 11)))
#days = ['2021-02-03','2021-02-04']
# days


# In[5]:


clients_devices = [
     #('Kcdda',    1, 0.004,     '5ea84272173b3d0011b3621f', '5ea9c267173b3d0011b36ee2'),
     #('SickKids', 1, 0.01,      '5e99a9e22386af0010ba7404', '5ec54b3d4d4a86001106cf08'),
    
    ('Guilford',1,0.01,'619807593896d10009e0b2ad','61981d0b9f156d000a8bf356'),
    #('Harrisonburg',1, 1/100,'6081cff532885f0009097bc3', '61f93abef66e8e0009967af5'),
    
    #('Coffee',1,1/100,'605a0334e3bda70008991d91','605a087f1bbdb60008c4cca1'),
    
    #('Williamsburg', 1, 1/100, '6082eaeb9295ea0008e39fe8', '6082ebef77cbe8000929680d'),
#     ('Urbandale','FW',1/100,'6195601bb469e60009ea7fce','61956106afe8280009b5a07b'),
    #('Saratoga','FW',1/100,'617861ac04d47b000952bb52','617862a1f449c80009ca3c84'),
    #new walla devices 
    ('Walla', 'VTPocketASA', 1/100, '60994789ee5d600009a40290', '6189947d22b29b0009b3b106'),
    ('Walla', 'VTCLDUOASA', 1/100, '60994789ee5d600009a40290', '618996f5f7b6af0009990482'),
    ('Walla', 1, 1/1000,      '60994789ee5d600009a40290', '609abe351175bd00090a92fd')

    
    #for the ipgroups
#         ('SickKids', 'SW1', 0.01,      '5e99a9e22386af0010ba7404', '5ec54b174d4a86001106cf03'),
#         ('SickKids', 'SW2', 0.01,      '5e99a9e22386af0010ba7404', '5ec54b884d4a8600110
    
#     ('Peachtree', 1, 0.01,     '5e85a5cc008b0c0010ed90da', '5f07251af684220011267b3f'),


    #('Rentpath','hq-fw', 1/100,      '60f96cecb56ac10008f635f2', '60ff96769fdd70000988ffc7'),
    #('Rentpath','dev-fw', 1/100,      '60f96cecb56ac10008f635f2', '60ff8a2d1ec3bc0008ca7454'),
    
    #('Krypto','FW', 1/10,      '6135c29aab9cc00009d19178', '6135cbd5b9719f0009250f69')
    
    #('HoustonCounty', 1, 1/50,      '603e5fa61788140011913529', '603e9f70be4cd50011efd064'),
    #('MorganTown', 1, 1/10,      '6040f84e99ef230011b54c81', '6040f8f199ef230011b54c9a')
    
    #('Roseville', 'FW-Main', 1/1000,      '607efa4be890af000920ea28', '607f089788a64c000921a432'),
    #('Roseville', 'FW-2', 1/1000,      '607efa4be890af000920ea28', '607f07e1557c4200098ca09c')
    
    #('Christiansburg','FW',1/1000,'606321174715f60008961b28','606324ea805d3a0008ba1072'),
    #('Palmetto',1,1/100,'607ee12ff2959700095a0ca7','60883934ee88920009f2454e')  
    
    #('Sioux', 1, 1/1000,      '60a52c5d7c0df300090597c2', '60a66eae1265d10009c77766'),
    #('JohnsonCity', 1, 1/1000,      '609e74f30bc7740009cd3fe0', '609e75ad2dbbae0009d94485'),
]


# In[ ]:


for client, device, sr, account_id, device_id in clients_devices:
    print(client, device, sr, account_id, device_id)
    for day in days:
        print(day)
        try:
            df = query_redshift(client, device, sr, day, account_id, device_id)
            tdf = df.groupby('time_window').agg({'flowPackets': 'sum'})
            #print(tdf.reindex(list(range(tdf.index.min(), tdf.index.max()+1, 5))).fillna(0).mean().values[0])
            print(np.mean(list(tdf['flowPackets'].values) + [0] * (17280 - len(tdf['flowPackets'].values))))
            print("less than a day: "+str(np.mean(list(tdf['flowPackets'].values))))
        except:
            pass


# In[ ]:


(df.time_window.max() - df.time_window.min())/5


# In[7]:


tdf.describe()


# In[6]:


import gzip
import boto3
import numpy as np
import pandas as pd

from io import BytesIO, TextIOWrapper
from sqlalchemy import create_engine
import psycopg2


# In[7]:


import gzip
import boto3
import numpy as np
import pandas as pd

from io import BytesIO, TextIOWrapper
from sqlalchemy import create_engine
import psycopg2

dbname='cynamics'
cluster='prodredshiftcluster.chk0levkctdj.us-east-1'
port=5439
# user='superadmin'
# password='usY45Np8SaV1^f9'
user='ro_user'
password='37gvWDy5GZnwwdXjhGgQ'



connstr = f'postgresql://{user}:{password}@{cluster}.redshift.amazonaws.com:5439/cynamics'
engine = create_engine(connstr) 
connection = psycopg2.connect(dbname=dbname, host=f'{cluster}.redshift.amazonaws.com', 
                              port='5439', user=user, password=password)

client, account_id, device_id = 'Eugene', '5ed4e7c5403dc20018b47e6c', '5f1b1cbd5a19bc0011aa66ff'
client, account_id, device_id = 'DouglasCounty', '5f15d4502997970011989705', '5f1eee725a19bc0011aa73dd'
client, account_id, device_id = 'WMU', '5ec7da694d4a86001106da83', '5ec7db437f736700114636a9'
client, account_id, device_id = 'Restorepoint', '5f53901a4e0ffa001a08ba2e', ''


# In[ ]:


start_time = 1597752000
column, value = 'sourceport', 443

query = f"""
WITH temp AS (
    SELECT extract(epoch from creationtime) - MOD(extract(epoch from creationtime), 5) as time_window, sourceport, SUM(flowpackets) as sum
    FROM public.rawdatas
    WHERE accountid='{account_id}'
      AND deviceid='{device_id}'
      AND extract(epoch from creationtime)>={start_time}
      AND extract(epoch from creationtime)<={start_time+3600}
    GROUP BY 1,2
)
SELECT sourceport, MAX(sum)
FROM temp
GROUP BY 1
"""

query = f"""
SELECT ipv4src, ipv4dest, SUM(flowpackets) as flowpackets
FROM public.rawdatas
WHERE accountid='{account_id}'
  AND deviceid='{device_id}'
  AND extract(epoch from creationtime)>={start_time}
  AND extract(epoch from creationtime)<={start_time}+3600
  AND {column}={value}
GROUP BY 1,2
"""

query = f"""
SELECT left(creationtime, 10), deviceid, samplingrate, count(*)
FROM public.rawdatas
WHERE accountid='{account_id}'
  ---AND deviceid='{device_id}'
GROUP BY 1,2,3
"""

# with connection.cursor() as cursor:
#     cursor.execute(query)
#     result = cursor.fetchall()

with engine.connect() as conn, conn.begin():
    query_df = pd.read_sql(query, conn)
query_df.sort_values(['deviceid', 'left'])


# In[ ]:


device_id


# In[ ]:


with connection.cursor() as cursor:
    cursor.execute(query)
    result = cursor.fetchall()
result


# In[ ]:


df[df.left == '2020-09-11']

df[['ipv4src', 'ipv4dest']] = np.sort(df[['ipv4src', 'ipv4dest']])
df.groupby(['ipv4src', 'ipv4dest']).sum().reset_index().sort_values('flowpackets', ascending=False)
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




