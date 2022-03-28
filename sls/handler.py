import json
import datetime

import time
import boto3
import os
import pandas as pd
# from pandas.compat import StringIO
from io import StringIO

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

def get_job_type(job_type):
    # complex processing
    job_type_processed = job_type
    return job_type_processed

def create_training_job(rows):
    
    print("In create training job")
     # For each input row in the JSON object...
    for row in rows:
        # Read the input row number (the output row number will be the same).
        
        # Read the first input parameter's value. For example, this can be a
        # numeric value or a string, or it can be a compound value such as
        # a JSON structure.
        _input_table_name = row[2]
        
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
            'InstanceType': os.environ['training_instance_type'],
            'InstanceCount': int(os.environ['training_instance_count']),
            'VolumeSizeInGB': int(os.environ['training_volume_size_gb']),
        },
        EnableManagedSpotTraining=True,
        StoppingCondition={
            'MaxRuntimeInSeconds': 10000,
            'MaxWaitTimeInSeconds': 10001
        }
    )

    training_job_arn = response['TrainingJobArn']

    return training_job_arn


def upload_to_s3(rows):
    # For each input row in the JSON object...
    body = ""
    for row in rows:
        #prefix = row[2]
        # extract and transform the user_ids and item_ids posted to csv
        body = body + row[2] + "," + row[3] + "\n"
    
    # print("body is ")
    # print(body)
    
    temp_filename = 'temp_file.csv'
    df = pd.DataFrame(StringIO(body), columns=['user_id', 'movie_id'])
    
    print("dataframe is....")
    print(df.head())
    df.to_csv(temp_filename)    
    
    bucket = os.environ['s3_bucket']                 
    data_prefix = os.environ['data_prefix']
    input_dir = os.environ['input_dir']
    
    prefix = os.path.join(data_prefix, input_dir)
    
    s3 = boto3.resource('s3')        
    
    response= s3.meta.client.upload_file(temp_filename, bucket, prefix + 'dataframe_for_scoring.csv')
    print(response)
    return 'successfully uploaded'
    
    
def create_deploy_job(rows):
    
    for row in rows:
        print(row)
        model_name = row[2]
        s3_model_uri = row[3]

    client = boto3.client('sagemaker')
    return _deploy_model(client, model_name, s3_model_uri)

def create_batch_job(rows):
    """
    
    """
# For each input row in the JSON object...
    for row in rows:
        model_name = row[2]
        s3_uri = row[3]
        # extract and transform the user_ids and item_ids posted to csv
            
    bucket = os.environ['s3_bucket'] 
    transform_jobname = "transform-job-" + time.strftime("%Y%m%d%H%M%S")
    
    output_dir = os.environ['output_dir']
    input_dir = os.environ['input_dir']
    if s3_uri=='default':
        s3_uri =  f's3://{bucket}/{input_dir}/'
    s3_output_location = f's3://{bucket}/{output_dir}/'
    
    client = boto3.client('sagemaker')
    
    response = client.create_transform_job(
                            TransformJobName=transform_jobname,
                            ModelName= model_name,
                            BatchStrategy='MultiRecord',
                            MaxConcurrentTransforms= 32,
                            MaxPayloadInMB= 100,
                            
                            TransformInput={
                                'DataSource': {
                                    'S3DataSource': {
                                        "S3DataType": "S3Prefix",                                        
                                        'S3Uri': s3_uri + 'dataframe_for_scoring.csv'
                                    }
                                },
                                'ContentType': 'text/csv', 
                                'SplitType': 'Line',
                                "CompressionType": "None",
                            },
                            TransformOutput={
                                'S3OutputPath': s3_output_location,
                                'AssembleWith': 'Line',
                                
                            },
                            TransformResources={
                                'InstanceType': os.environ['transform_instance_type'],
                                'InstanceCount': int(os.environ['transform_instance_count']),                                
                            },
                )
                
    print(f'predictions are in location: {s3_output_location}')
    modelarn =  response['ModelArn']
    return modelarn

def _deploy_model(client, model_name, s3_model_uri):
    
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
    return modelarn

def aws_job(event, context):

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
        
        job_arn= ""
        
        job_type = get_job_type(rows[0][1])
        print("Job type is ", job_type)

        if job_type=='training':
            job_arn = create_training_job(rows)
        elif job_type=='deploy':
            job_arn = create_deploy_job(rows)
        elif job_type=='upload_to_s3':
            upload_to_s3(rows)
        elif job_type=='batch_job':
            job_arn= create_batch_job(rows)
        
        array_of_rows_to_return = []
        # Put the returned row number and the returned value into an array.
        row_to_return = [0, job_arn]

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



