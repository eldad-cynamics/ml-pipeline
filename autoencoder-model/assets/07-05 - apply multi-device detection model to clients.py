#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.append('..')
sys.path.append('../03 inference')
from inference_utils import *
from generic_detector_utils import *


# # Load detection model

# In[2]:


multi_device_detection_model = load_tuned_model(3)


# # Load data
clients_devices = [
#     ('Kcdda', 1, 0.004),
#     ('SickKids', 1, 0.01),
#     ('WMU', 1, '0.00002'),
#     ('Peachtree', 1, 0.01),
#     ('Eugene', 'Firewall', 0.0005),
#     ('Eugene', 1, 0.0005),
#     ('Eugene', 2, 0.0005),
#     ('DouglasCounty', 1, 0.000122),
#    ('Restorepoint', 1, 0.1),
#       ('Eugene','Firewall',0.0005)
#('Underline',1,0.03125)
#('Allegronet',1,0.01)
#('Sheba',1,0.02)
#('Sheba',2,0.5)
#('NPS','DCFW',0.01)

    #('MorganTown',1,0.1)
    #('HoustonCounty', 1, 1/50)
    
    ('Clakamas', 1, 1/50)
    
    #('Hamblen', 'FWCH', 0.1,1),
    #('Hamblen', 'FWJC', 0.1,1)
]
client, device, sr = clients_devices[0]
model_version, account_id, device_id, v1_model_name, v3_model_name, _ = get_device_details(client, device, sr)
client, device, sr, model_version, account_id, device_id, v1_model_name, v3_model_name
# In[3]:


# client, devices = 'Fayette',  [
#     (182, 0.001, 1, 'sagemaker-ecr-2020-06-28-15-29-44-554', 'sagemaker-ecr-2020-06-28-15-33-13-393'),
#     (228, 0.001, 1, 'sagemaker-ecr-2020-06-28-15-06-49-985', 'sagemaker-ecr-2020-06-28-15-34-44-518')
# ]

#client, devices = 'Kcdda', [(1, 0.004, 1, '06-24-Kcdda-1-v1-048-f9076f00', '06-24-Kcdda-1-v3-001-ec8de1a5')]
#client, devices = 'Kcdda', [(1, 0.004, 2, '04-12-Kcdda-1-v1-002-830544d0','04-12-Kcdda-1-v3-009-4162de2f')]


#client, devices = 'SickKids', [(1, 0.01, 6, '06-13-SickKids-1-v1-016-a6e6643f','06-13-SickKids-1-v3-019-7b79f409')]
#client, devices = 'Harrisonburg', [(1,1/1000,2,'10-03-Harrisonburg-1-v1-010-1ccaabef','10-03-Harrisonburg-1-v3-022-cc67c44a')]



#client, devices ='Roseville', [('FW-Main',1/1000,2,'06-24-Roseville-FW-Main-v1-028-65a05e34','06-24-Roseville-FW-Main-v3-013-adf1a6ed')],
#client, devices ='Roseville',[('FW-2',1/1000,2,'06-24-fw2-Roseville-FW-2-v1-032-75bc498e','06-24-fw2-Roseville-FW-2-v3-034-5523cf38')]

#client, devices ='Roseville', [('FW-Main',1/1000,2,'06-24-Roseville-FW-Main-v1-028-65a05e34','06-24-Roseville-FW-Main-v3-013-adf1a6ed'),
#                              ('FW-2',1/1000,2,'06-24-fw2-Roseville-FW-2-v1-032-75bc498e','06-24-fw2-Roseville-FW-2-v3-034-5523cf38')]


#client, devices = 'Sioux', [(1,1/1000,1,'06-22-Sioux-1-v1-029-9d0b5553','06-22-Sioux-1-v3-010-4616c69a')]
#client, devices = 'JohnsonCity', [(1,1/1000,1,'06-21-JohnsonCity-1-v1-014-fa685b46','06-21-JohnsonCity-1-v3-002-14de8a60')]
#client,devices = 'Walla',[(1,1/1000,1,'06-16-Walla-1-v1-006-fc264554','06-16-Walla-1-v3-007-d10964ba')]

#client, devices = 'HoustonCounty', [(1,0.02,2,'07-27-HoustonCounty-1-v1-036-e10753d0','07-27-HoustonCounty-1-v3-016-0e9d776f')]   
#client, devices = 'MorganTown', [(1,0.1,1,'04-26-M2-MorganTown-1-v1-018-2fd0a21e','03-09-MORG-MorganTown-1-v3-034-91e0c72e')]
#client, devices = 'Williamsburg',[(1,1/100,2,'10-03-Williamsburg-1-v1-010-e05b6d09','10-03-Williamsburg-1-v3-031-d194631c')]

#client, devices = 'Williamsburg',[(1,1/100,3,'12-26-Williamsburg-1-v1-011-4f5a4f4f','12-26-Williamsburg-1-v3-025-84ee8c2b')]
#client, devices = 'Urbandale',[('FW',1/100,1,'12-26-Urbandale-FW-v1-011-e23e71d8','12-26-Urbandale-FW-v3-008-d3e54ac1')]
#client, devices = 'Saratoga',[('FW',1/100,1,'12-26-Saratoga-FW-v1-007-94a5e215','12-26-Saratoga-FW-v3-006-680275e3')]


#client,devices = 'Coffee', [(1,0.01,4,'10-03-Coffee-1-v1-001-56768a1f','10-03-Coffee-1-v3-006-a836e8bb')]
#client,devices  = 'Palmetto',[(1,1/100,1,'05-20-Palmetto-1-v1-029-4f48d948','05-20-Palmetto-1-v3-011-131a28a4')]
#client, devices = 'Christiansburg', [('FW',0.001,1,'04-11-Christiansburg-FW-v1-013-8af25341','03-30-Clarks-Clarksville-FW-v3-014-015ee306')]

client, devices = 'Guilford', [
    (1,0.01,1,'03-16-Guilford-1-v1-003-336da5d6','03-16-Guilford-1-v3-007-5ab0ca39')]

#client, devices = 'Walla', [
#    (1,1/1000,1,'06-16-Walla-1-v1-006-fc264554','06-16-Walla-1-v3-007-d10964ba'),
#    ('VTPocketASA', 1/100, 1, '12-26-Walla-VTPocketASA-v1-010-ad4072d6','12-26-Walla-VTPocketASA-v3-009-251d720c'),
#    ('VTCLDUOASA', 1/100, 1, '12-26-Walla-VTCLDUOASA-v1-026-29f2d7bd','12-26-Walla-VTCLDUOASA-v3-030-1d2f638c'),
#]


# client, devices = 'Peachtree', [
#     #(1, 0.01, 1, '07-19-Peachtree-1-v1-036-be6efcbb', '07-19-Peachtree-1-v3-034-5d30a482'),
#     #(1, 0.01, 2, 'sagemaker-ecr-2020-07-22-12-10-13-157', 'sagemaker-ecr-2020-07-22-13-16-57-398'),
#     (1, 0.01, 3, '07-25-Peachtree-1-v1-018-db3d263c', '07-25-Peachtree-1-v3-032-ed8e6f89'),
#     #(1, 0.01, 4, 'sagemaker-ecr-2020-08-02-10-06-32-750', 'sagemaker-ecr-2020-08-02-10-05-29-231-copy-08-03'),
# ]

#client, devices = 'Underline', [
#    (1,0.03125,1,'09-23-aviv-under-Underline-1-v1-017-422c111f','09-23-aviv-under-Underline-1-v3-012-ee30434a')
#]


#update days!
labels_dict, loss_dict = read_data_into_dictionaries(client, devices, days=['2022-02-17',
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
 '2022-03-14'])


# In[4]:


devices


# # Infer

# In[5]:


for day in loss_dict.keys():
    if day == 'training' or day < '2020-01-10':
        continue
    analyze_day(client, devices, day, loss_dict, labels_dict, multi_device_detection_model,
                #plot=False
               )


# # playground

# In[ ]:


device, sr, model_version, v1_model, v3_model = devices[0]
device, sr, model_version, v1_model, v3_model


# In[ ]:


for i in range(18):
    plt.figure(figsize=(16,9))
    plt.plot(loss_dict['2020-07-16'][device][:,i])
    plt.title(i)
    plt.show()


# In[ ]:


v3_loss_filepath = f'clients/{client}/sr={sr}/device={device}/version=3/model={model_version}/{v3_model}'
boto3.resource('s3').Bucket('rsrch-cynamics-datasets').download_file(f'{v3_loss_filepath}/{day}.loss', 'temp_file.h5')
with h5py.File('temp_file.h5', 'r') as h5f:
    v3_loss = h5f['loss'][:]


# In[ ]:


i


# In[ ]:


for feature in ['sourcePort_443', 'destPort_443']:
    i = get_features_names().index(feature)
    plt.figure(figsize=(16,9))
    plt.plot(v3_loss[:,i])
    plt.title(f'{day}, device={device}, {feature}')
    plt.show()


# # Map time window to actual time

# In[ ]:


time_window_to_timestamp('2020-07-16', 9902)
time_window_to_timestamp('2020-07-16', 9956)


# In[ ]:


import pandas as pd
def time_window_to_timestamp(day, time_window):
    hour = int(time_window/17280*24)
    minute = int(time_window/17280*24%1*60)
    second = int(time_window/17280*24%1*60%1*60)
    time_str = f'{day} {hour}:{minute}:{second}'
    print((pd.to_datetime(time_str) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'), time_str)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import h5py
from sklearn.metrics import roc_auc_score, roc_curve
with h5py.File('../05 tune generic detector/test.h5', 'r') as h5f:
    X_test = h5f['X'][:]
    y_test = h5f['y'][:]


# In[ ]:


for i in range(50):
    model = load_new_tuned_model(i)
#     model = load_model('detection_model_phase1.h5')

    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test.reshape(-1), y_pred.reshape(-1))
    roc_auc = roc_auc_score(y_test.reshape(-1), y_pred.reshape(-1))
    print(f'auc: {roc_auc}')

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc
            )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Trained Model {i}')
    plt.legend(loc="lower right")
    plt.show()
    break


# In[ ]:


y_test.any(axis=1).shape, (y_pred>0.5).any(axis=1).shape


# In[ ]:


y_pred


# In[ ]:





# In[ ]:


from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test.any(axis=1), (y_pred>0.5).any(axis=1)).ravel()


# In[ ]:


tn, fp, fn, tp


# In[ ]:




