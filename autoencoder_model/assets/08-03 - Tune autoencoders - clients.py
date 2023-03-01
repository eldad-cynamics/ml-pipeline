#!/usr/bin/env python
# coding: utf-8

# ## Set up the notebook instance to support local mode
# Currently you need to install docker-compose in order to use local mode (i.e., testing the container in the notebook instance without pushing it to ECR).

# In[1]:


get_ipython().system('/bin/bash setup.sh')


# ## Set up the environment
# We will set up a few things before starting the workflow. 
# 
# 1. get the execution role which will be passed to sagemaker for accessing your resources such as s3 bucket
# 2. specify the s3 bucket and prefix where training data set and model artifacts are stored

# In[2]:


import tensorflow as tf

import os
import shlex
import boto3
import tarfile
import sagemaker
import subprocess
from time import gmtime, strftime
from sagemaker.estimator import Estimator

from tuning_utils import *

region = boto3.Session().region_name

sagemaker_session = sagemaker.Session()
smclient = boto3.client('sagemaker')

repository = 'sagemaker-ecr'
bucket = 'rsrch-cynamics-datasets'
prefix = 'autoencoders'
tensorflow_version = '2.1.0-py3'


role = sagemaker.get_execution_role()

#fixex attempts below - restart notebook solved it
#iam = boto3.client('iam')
#role = iam.get_role(RoleName='arn:aws:iam::662612070855:role/service-role/AmazonSageMaker-ExecutionRole-20200303T115551')['Role']['Arn']
#try:
#    role = sagemaker.get_execution_role()
#except ValueError:
#iam = boto3.client('iam')
#role = iam.get_role(RoleName='AmazonSageMakerFullAccess')['Role']['Arn']
#from sagemaker import get_execution_role
#sagemaker_session = sagemaker.Session()
#role = get_execution_role()


# ## Prepare the data

# In[3]:


devices_models = [
#     ('Kcdda', 1, 0.004, 2),

#     ('SickKids', 1, 0.01, 1),
     #('SickKids', 1, 0.01, 6),
#         ('TLpeach', 1, 1, 1)

#     ('WMU', 1, '0.00002', 1),

#     ('Peachtree', 1, 0.01, 1),
#     ('Peachtree', 1, 0.01, 2),
#     ('Peachtree', 1, 0.01, 3),
#     ('Peachtree', 1, 0.01, 4),
#     ('Peachtree', 1, 0.01, 5),

#     ('Eugene', 'Firewall', 0.0005, 2),
#     ('Eugene', 1, 0.0005, 1),
#     ('Eugene', 2, 0.0005, 1),
    
#     ('DouglasCounty', 1, 0.0001220703125, 1),
    
#     ('Restorepoint', 1, 0.1, 1),
#    ('Restorepoint', 1, 0.1, 2),
#        ('Underline',1,0.03125,1)
#    ('Allegronet',1,0.01,1)
#    ('Sheba',1,0.02,1)
#        ('Sheba',2,0.5,1)
#        ('NPS', 'DCFW', 0.01,1)
    
     #    ('HoustonCounty', 1, 1/50,  2),
    #('MorganTown', 1, 1/10,1)
    
    #('Roseville', 'FW-Main', 1/1000,2),
    #('Roseville', 'FW-2', 1/1000,2)  
    
    #('Harrisonburg',1, 1/1000,2)
    
        #('Hamblen', 'FWCH', 0.1,1),
#    ('Hamblen', 'FWJC', 0.1,1)
    #('Clakamas', 1, 1/50,1)
    #('Fonex','FW1',1/50,1)
    #('Fonex','FW-Fr',1/50,1)
    
    #('Clarksville','FW',1/1000,1)
    #('Coffee',1,1/100,4)
    
    #('Sioux', 1, 1/1000,1)
    #('JohnsonCity', 1, 1/1000,1)
    #('Walla', 1, 1/1000,1)
    #('Christiansburg','FW',1/1000,1)
    
    #('Williamsburg', 1, 1/100, 3)
    #('Urbandale','FW',1/100,1),
    #('Saratoga','FW',1/100,1),
    ('Guilford',1,0.01,1),
    #new walla devices 
    #('Walla', 'VTPocketASA', 1/100, 1),
    #('Walla', 'VTCLDUOASA', 1/100, 1)

    
    #('Palmetto',1,1/100,1)
    #('Franklin',1,1/100,1)

]

features_version = 1
client, device, sr, model_version = devices_models[-1]
print(client, device, sr, model_version, f'v{features_version}')

channels = {
    'train': f's3://rsrch-cynamics-datasets/clients/{client}/sr={sr}/device={device}/version={features_version}/model={model_version}/type=train/',
    'val': f's3://rsrch-cynamics-datasets/clients/{client}/sr={sr}/device={device}/version={features_version}/model={model_version}/type=val/',
}
print(channels)

hyperparameters = {'beta1': '0.9995694616075936',
                   'beta2': '0.9998864470941636',
                   'dropout': '0.044423240350256694',
                   'learning_rate': '0.00001',
                   'epochs': '500',
                  }
hyperparameters


# ## Building the image
# We will build the docker image using the Tensorflow versions on dockerhub. The full list of Tensorflow versions can be found at https://hub.docker.com/r/tensorflow/tensorflow/tags/
# 
# 
# ## Pushing the container to ECR
# Now that we've tested the container locally and it works fine, we can move on to run the hyperparmeter tuning. Before kicking off the tuning job, you need to push the docker image to ECR first. 
# 
# The cell below will create the ECR repository, if it does not exist yet, and push the image to ECR.

# In[4]:


get_ipython().run_cell_magic('time', '', "\ndef build_image(name, version):\n    cmd = 'docker build -t %s --build-arg VERSION=%s -f Dockerfile .' % (name, version)\n    subprocess.check_call(shlex.split(cmd))\n\naccount = boto3.client('sts').get_caller_identity()['Account']\n\nimage_name = f'{account}.dkr.ecr.{region}.amazonaws.com/{repository}:{prefix}'\n\nprint('building image:'+image_name, end=' ')\nbuild_image(image_name, tensorflow_version)\nprint('Done!')\n\n# # If the repository doesn't exist in ECR, create it.\n# exist_repo = !aws ecr describe-repositories --repository-names {repository} > /dev/null 2>&1\n\n# if not exist_repo:\n#     print('Creating')\n#     !aws ecr create-repository --repository-name {repository} > /dev/null\n\n# Get the login command from ECR and execute it directly\n!$(aws ecr get-login --region {region} --no-include-email)\n\n!docker push {image_name}\n")


# In[ ]:


### Create a training job using local mode
output_location = f's3://{bucket}/{prefix}/{client}-{device}-v{features_version}/'
estimator = Estimator(image_name,
                      role=role,
                      output_path=output_location,
                      train_instance_count=1,
#                       train_instance_type='local',
                      train_instance_type='ml.m5.xlarge' if features_version == 1 else 'ml.m5.2xlarge',
                      hyperparameters=hyperparameters)
estimator.fit(channels)


# # Run experiment on all combinations

# In[5]:


training_image = image_name
test = False
global_jobs = '03-16' + ('-test' if test else '')

for features_version in [1,3]:
    for client, device, sr, model_version in devices_models:
        tuning_job_name = f'{global_jobs}-{client}-{device}-v{features_version}'
        channels = {
            'train': f's3://rsrch-cynamics-datasets/clients/{client}/sr={sr}/device={device}/version={features_version}/model={model_version}/type=train/',
            'val': f's3://rsrch-cynamics-datasets/clients/{client}/sr={sr}/device={device}/version={features_version}/model={model_version}/type=val/',
        }
        tuning_job_config = get_config(max_jobs=1 if test else 50, max_parallel=2)
        training_job_definition = get_definition(training_image, channels, bucket, 
                                                 global_jobs, tuning_job_name, 1 if test else 500,
                                                 client, device, sr, role)
        training_job_definition['StaticHyperParameters']['model_version'] = str(model_version)
        training_job_definition['StaticHyperParameters']['features_version'] = str(features_version)
        training_job_definition['ResourceConfig']['InstanceType'] = 'ml.m5.xlarge' if features_version == 1 else 'ml.m5.2xlarge'
        #display(training_job_definition['OutputDataConfig'])
        
        try:
            output = smclient.create_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name,
                                                                HyperParameterTuningJobConfig=tuning_job_config,
                                                                TrainingJobDefinition=training_job_definition,
                                                               )

            status = smclient.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)['HyperParameterTuningJobStatus']
            display(status)
            if status =='InProgress':
                upload_code(training_job_definition['OutputDataConfig']['S3OutputPath'])
        except Exception as e:
            print(e)


# In[ ]:




