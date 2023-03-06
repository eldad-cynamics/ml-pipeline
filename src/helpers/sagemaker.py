import boto3
import sagemaker
from loguru import logger
from typing import Dict
from sagemaker import get_execution_role

def stop_sagemaker_notebook_instance (notebook_name: str) -> None:
    sm = boto3.client('sagemaker')
    sm.stop_notebook_instance(NotebookInstanceName=f"{notebook_name}")

def get_sagemaker_defualt_bucket_name():
    try: 
        role = get_execution_role()
    except Exception as exc:
        logger.info(f"An exception occurred when trying to get the execution role: {exc}")
        raise exc
    
    s3 = boto3.resource('s3')
    defualt_bucket_name = 'sagemaker-' + boto3.Session().region_name + '-' + boto3.client('sts').get_caller_identity()['Account']
    return defualt_bucket_name
    
    
def get_best_sagemaker_tuning_job_results (tuning_job_name: str) -> Dict:
    
    sagemaker_client = boto3.client('sagemaker')

    try:

        tuning_job_description = sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name)

        best_training_job_name = tuning_job_description['BestTrainingJob']['TrainingJobName']

        best_training_job_description = sagemaker_client.describe_training_job(
            TrainingJobName=best_training_job_name)

        best_training_job_hyperparameters = best_training_job_description['HyperParameters']

        logger.info(f"Best training job name: {best_training_job_name}")
        logger.info(f'Hyperparameters for the best training job: {best_training_job_hyperparameters}')

        tuning_job_results = {
            'best_training_job_name' : best_training_job_name,
            'best_training_job_hyperparameters': best_training_job_hyperparameters

        }

    except Exception as exc:
        logger.exception(f"Can't fetch results from sagemaker for tuning job: {tuning_job_name}")
        tuning_job_results = None
        
    return tuning_job_results

if __name__ == '__main__':

    tuning_job_results = get_best_sagemaker_tuning_job_results(tuning_job_name='test-eldad-Sioux-1-v2')