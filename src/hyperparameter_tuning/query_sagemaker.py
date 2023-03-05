import boto3
import sagemaker
from loguru import logger
from typing import Dict


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
