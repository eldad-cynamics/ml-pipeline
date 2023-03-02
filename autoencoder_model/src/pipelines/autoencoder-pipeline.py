import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor

region = boto3.Session().region_name

sagemaker_session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker')
role = sagemaker.get_execution_role()


# Define your preprocessing scripts
preprocess_script_1 = "preprocess_1.py"
preprocess_script_2 = "preprocess_2.py"


# Define your preprocessing step 1
preprocess_step_1 = ProcessingStep(
    name="PreprocessStep1",
    processor=ScriptProcessor(
        image_uri="public.ecr.aws/sagemaker/python:3.7",
        command=["python3"],
        instance_count=1,
        instance_type="ml.t2.2xlarge",
        role=role
    ),
    inputs=[
        sagemaker.processing.ProcessingInput(
            source="s3://your-bucket/data",
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            source="/opt/ml/processing/output",
            destination="s3://your-bucket/preprocessed_data_1"
        )
    ],
    script=preprocess_script_1
)

# Define your preprocessing step 2
preprocess_step_2 = ProcessingStep(
    name="PreprocessStep2",
    processor=ScriptProcessor(
        image_uri="your-ecr-image-uri",
        command=["python3"],
        instance_count=1,
        instance_type="ml.m5.large",
        role="your-sagemaker-role"
    ),
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=preprocess_step_1.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            source="/opt/ml/processing/output",
            destination="s3://your-bucket/preprocessed_data_2"
        )
    ],
    script=preprocess_script_2
)

# Define your pipeline
pipeline = Pipeline(
    name="MyPipeline",
    parameters=[],
    steps=[preprocess_step_1, preprocess_step_2],
    sagemaker_session=sagemaker.Session()
)

# Create and run your pipeline
pipeline.upsert(role_arn="your-sagemaker-role-arn")
execution = pipeline.start()

if __name__ == '__main__':
    logger.info(f'{get_current_date_and_time()}')




