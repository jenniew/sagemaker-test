#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

chmod +x logistic_regression/train
chmod +x logistic_regression/serve

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

sudo docker build \
--build-arg http_proxy=http://proxy.sc.intel.com:911 \
--build-arg https_proxy=https://proxy.sc.intel.com:911 \
--rm -t ${image} .
#docker tag ${image} ${fullname}
