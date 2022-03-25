import json
import datetime
from multiprocessing.connection import wait
import time
import boto3
import os

def wait_for_job_to_complete(client, job_name, job_type='training'):
    if job_type == 'training':
        waiter = client.get_waiter('training_job_completed_or_stopped')
        waiter.wait(
            TrainingJobName=job_name,
            WaiterConfig={
                'Delay': 123,
                'MaxAttempts': 123
            }
        )
    elif job_type == 'transform':        
        waiter = client.get_waiter('transform_job_completed_or_stopped')
        waiter.wait(
            TransformJobName=job_name,
            WaiterConfig={
                'Delay': 123,
                'MaxAttempts': 123
            }
        )        
def train_and_deploy(event, context):

    # 200 is the HTTP status code for "ok".
    status_code = 200

    try:
        # From the input parameter named "event", get the body, which contains
        # the input rows.
        event_body = event["body"]

        # Convert the input from a JSON string into a JSON object.
        payload = json.loads(event_body)
        # This is basically an array of arrays. The inner array contains the
        # row number, and a value for each parameter passed to the function.
        rows = payload["data"]

        # For each input row in the JSON object...
        for row in rows:
            # Read the input row number (the output row number will be the same).
            row_number = row[0]

            # Read the first input parameter's value. For example, this can be a
            # numeric value or a string, or it can be a compound value such as
            # a JSON structure.
            _input_table_name = row[1]
            model_name = row[2]
            mode = row[3]


        # start the SageMaker training job
        client = boto3.client('sagemaker')

        bucket = os.environ['s3_bucket'] 
        prefix = "training-job-" + time.strftime("%Y%m%d%H%M%S")

        s3_output_location = 's3://{}/'.format(bucket)

        print(s3_output_location)

        training_job_name = prefix
        TRAINING_IMAGE_ECR_PATH = os.environ['training_image_ecr_path']
        SAGEMAKER_ROLE_ARN = os.environ['sagemaker_role_arn']
        

        response = client.create_training_job(
            TrainingJobName=training_job_name,
            HyperParameters=dict(input_table_name=_input_table_name, region=os.environ['region']),
            AlgorithmSpecification={
                'TrainingImage': TRAINING_IMAGE_ECR_PATH,
                'TrainingInputMode': 'File'
            },
            RoleArn=SAGEMAKER_ROLE_ARN,         
            OutputDataConfig={
                'S3OutputPath': s3_output_location
            },
            ResourceConfig={
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 10,
            },
            EnableManagedSpotTraining=True,
            StoppingCondition={
                'MaxRuntimeInSeconds': 10000,
                'MaxWaitTimeInSeconds': 10001
            }
        )

        training_job_arn = response['TrainingJobArn']
        print(training_job_arn)

        wait_for_job_to_complete(client, training_job_name, job_type='training')        
        print("Training completed")
        
        print("Creating the model")
        
        s3_model_uri = os.path.join(s3_output_location, training_job_name,'output', 'model.tar.gz')

        deploy_model(client, model_name, s3_model_uri, mode)

        array_of_rows_to_return = []
        # Put the returned row number and the returned value into an array.
        row_to_return = [0, training_job_arn]

        # ... and add that array to the main array.
        array_of_rows_to_return.append(row_to_return)

        json_compatible_string_to_return = json.dumps({"data" : array_of_rows_to_return})

    except Exception as err:
        # 400 implies some type of error.
        status_code = 400
        # Tell caller what this function could not handle.
        print(err)
        json_compatible_string_to_return = str(err)

    # Return the return value and HTTP status code.
    return {
        'statusCode': status_code,
        'body': json_compatible_string_to_return
    }


def deploy_model(client, model_name, s3_model_uri, mode='batch_mode'):

    # 200 is the HTTP status code for "ok".
    status_code = 200

    try:
        
        # start the SageMaker training job
        client = boto3.client('sagemaker')

        ECR_PATH = os.environ['training_image_ecr_path'] 
        SAGEMAKER_ROLE_ARN = os.environ['sagemaker_role_arn']

        response = client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': ECR_PATH,
                'ModelDataUrl': s3_model_uri
            },
            ExecutionRoleArn=SAGEMAKER_ROLE_ARN
        )

        modelarn =  response['ModelArn']
        array_of_rows_to_return = []
        # Put the returned row number and the returned value into an array.
        row_to_return = [0, modelarn]

        # ... and add that array to the main array.
        array_of_rows_to_return.append(row_to_return)

        json_compatible_string_to_return = json.dumps({"data" : array_of_rows_to_return})

    except Exception as err:
        # 400 implies some type of error.
        status_code = 400
        # Tell caller what this function could not handle.
        print(err)
        json_compatible_string_to_return = str(err)

    # Return the return value and HTTP status code.
    return {
        'statusCode': status_code,
        'body': json_compatible_string_to_return
    }


# function that performs real-time prediction
def generate_batch_predictions(event, context):

    # 200 is the HTTP status code for "ok".
    status_code = 200

    try:
        # From the input parameter named "event", get the body, which contains
        # the input rows.
        event_body = event["body"]

        # Convert the input from a JSON string into a JSON object.
        payload = json.loads(event_body)
        # This is basically an array of arrays. The inner array contains the
        # row number, and a value for each parameter passed to the function.
        rows = payload["data"]

        # For each input row in the JSON object...
        body = ""
        for row in rows:
            model_name = row[1]
            # extract and transform the user_ids and item_ids posted to csv
            body = body + row[2] + "," + row[3] + "\n"

        import pandas as pd
        temp_filename = 'temp_file.csv'
        pd.DataFrame(body).to_csv(temp_filename)    
        
        # invoke the SageMaker endpoint
        client = boto3.client('sagemaker')
        bucket = os.environ['s3_bucket'] 
        transform_jobname = "transform-job-" + time.strftime("%Y%m%d%H%M%S")
        
        local_directory = 'data'
        prefix          = '/input'

        s3 = boto3.resource('s3')

        s3_input_location =  f's3://{bucket}/input/'
        s3_output_location = f's3://{bucket}/predictions/'
        
        s3.meta.client.upload_file(temp_filename, bucket, s3_input_location + 'dataframe_for_scoring.csv')

        print(s3_output_location)

        response = client.create_transform_job(
                            TransformJobName=transform_jobname,
                            ModelName= model_name,
                            BatchStrategy='MultiRecord',
                            Environment={
                                'string': 'string'
                            },
                            TransformInput={
                                'DataSource': {
                                    'S3DataSource': {                                        
                                        'S3Uri': s3_input_location + 'dataframe_for_scoring.csv'
                                    }
                                },
                                'ContentType': 'text/csv', 
                                'SplitType': 'Line'
                            },
                            TransformOutput={
                                'S3OutputPath': s3_output_location,
                                'AssembleWith': 'Line',
                                
                            },
                            TransformResources={
                                'InstanceType': 'ml.m3.medium',
                                'InstanceCount': 123,
                                'VolumeKmsKeyId': 'string'
                            },
                )
        
        wait_for_job_to_complete(client, transform_jobname, job_type='transform')

        modelarn =  response['ModelArn']
        array_of_rows_to_return = []
        # Put the returned row number and the returned value into an array.
        row_to_return = [0, modelarn]

        # ... and add that array to the main array.
        array_of_rows_to_return.append(row_to_return)

        # json_compatible_string_to_return = json.dumps({"data" : array_of_rows_to_return})

        # i = 0
        # array_of_rows_to_return = []
        
        # predictions = response["TransformJobArn"]

        # for prediction in iter(predictions.splitlines()):
        #     # Put the returned row number and the returned value into an array.
        #     row_to_return = [i, prediction]
        #     # ... and add that array to the main array.
        #     array_of_rows_to_return.append(row_to_return)
        #     i = i + 1

        json_compatible_string_to_return = json.dumps({"data" : array_of_rows_to_return})

    except Exception as err:
        # 400 implies some type of error.
        status_code = 400
        # Tell caller what this function could not handle.
        print(err)
        json_compatible_string_to_return = str(err)

    # Return the return value and HTTP status code.
    return {
        'statusCode': status_code,
        'body': json_compatible_string_to_return
    }