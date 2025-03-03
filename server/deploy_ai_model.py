import os
import sys
import boto3
from botocore.exceptions import ClientError

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

S3_BUCKET = os.getenv('S3_BUCKET_NAME')
MODEL_ARTIFACTS_PATH = os.getenv('MODEL_ARTIFACTS_PATH', 'think-ai-output/model.tar.gz')
MODEL_NAME = os.getenv('SAGEMAKER_MODEL_NAME', 'think-ai-model')
ENDPOINT_CONFIG_NAME = os.getenv('SAGEMAKER_ENDPOINT_CONFIG_NAME', 'think-ai-endpoint-config')
ENDPOINT_NAME = os.getenv('SAGEMAKER_ENDPOINT_NAME', 'think-ai-endpoint')
ROLE_ARN = os.getenv('SAGEMAKER_ROLE_ARN')

sagemaker_client = boto3.client('sagemaker', region_name='us-west-2')
s3_client = boto3.client('s3', region_name='us-west-2')

def upload_model_to_s3():
    try:
        print("Uploading model artifacts to S3...")
        s3_client.upload_file(MODEL_ARTIFACTS_PATH, S3_BUCKET, 'think-ai-output/model.tar.gz')
        print("Model artifacts uploaded to S3.")
    except ClientError as e:
        print(f"Failed to upload model to S3: {e}")
        sys.exit(1)

def create_or_update_model():
    try:
        print("Creating or updating SageMaker model...")
        response = sagemaker_client.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={
                'Image': '382416733822.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.3.0',  # Update as needed
                'ModelDataUrl': f's3://{S3_BUCKET}/think-ai-output/model.tar.gz',
            },
            ExecutionRoleArn=ROLE_ARN
        )
        print("SageMaker model created/updated.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUse':
            print("Model already exists. Updating the model...")
            sagemaker_client.delete_model(ModelName=MODEL_NAME)
            sagemaker_client.create_model(
                ModelName=MODEL_NAME,
                PrimaryContainer={
                    'Image': '382416733822.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.3.0',  # Update as needed
                    'ModelDataUrl': f's3://{S3_BUCKET}/think-ai-output/model.tar.gz',
                },
                ExecutionRoleArn=ROLE_ARN
            )
            print("SageMaker model updated.")
        else:
            print(f"Failed to create/update model: {e}")
            sys.exit(1)

def create_endpoint_config():
    try:
        print("Creating SageMaker endpoint configuration...")
        response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': MODEL_NAME,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'InitialVariantWeight': 1
                },
            ]
        )
        print("Endpoint configuration created.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUse':
            print("Endpoint configuration already exists. Updating the configuration...")
            sagemaker_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
            sagemaker_client.create_endpoint_config(
                EndpointConfigName=ENDPOINT_CONFIG_NAME,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': MODEL_NAME,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.m5.large',
                        'InitialVariantWeight': 1
                    },
                ]
            )
            print("Endpoint configuration updated.")
        else:
            print(f"Failed to create/update endpoint configuration: {e}")
            sys.exit(1)

def create_or_update_endpoint():
    try:
        print("Creating SageMaker endpoint...")
        response = sagemaker_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME
        )
        print("Endpoint creation initiated. Waiting for endpoint to be in service...")
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=ENDPOINT_NAME)
        print("Endpoint is in service.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUse':
            print("Endpoint already exists. Updating the endpoint configuration...")
            sagemaker_client.update_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=ENDPOINT_CONFIG_NAME
            )
            print("Endpoint configuration updated. Waiting for endpoint to be in service...")
            waiter = sagemaker_client.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=ENDPOINT_NAME)
            print("Endpoint is in service.")
        else:
            print(f"Failed to create/update endpoint: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred while creating/updating the endpoint: {e}")
        sys.exit(1)

def main():
    upload_model_to_s3()
    create_or_update_model()
    create_endpoint_config()
    create_or_update_endpoint()
    print("AI model deployed successfully.")

if __name__ == "__main__":
    main()