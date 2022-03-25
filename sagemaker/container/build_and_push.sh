#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
aws_role=$2

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

if [ "$aws_role" == "" ]
then    
    aws_role=default
    echo "Role is set to: $aws_role"
fi

chmod +x recommendation_engine/train
chmod +x recommendation_engine/serve

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text --profile ${aws_role})
echo $account


if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" --profile "${aws_role}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" --profile "${aws_role}" > /dev/null
fi

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region "${region}" --profile "${aws_role}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.


docker buildx build -o type=docker --platform=linux/amd64 -t ${image} .
docker tag ${image} ${fullname}

docker push ${fullname}