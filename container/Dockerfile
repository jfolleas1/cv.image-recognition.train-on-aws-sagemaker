# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# For more information on creating a Dockerfile
# https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile
FROM tensorflow/tensorflow:2.2.0rc2-gpu-py3-jupyter

# RUN apt-get update && apt-get install -y --no-install-recommends nginx curl

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /modeling /opt/ml/code
WORKDIR /opt/ml/code

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade sagemaker-training 
RUN pip install -r requirements.txt

ENV MODEL_PATH=/opt/ml/model
ENV INPUT_PATH=/opt/ml/input

RUN ln -s $INPUT_PATH/data $CODE_PATH/data
RUN ln -s $MODEL_PATH $CODE_PATH/output

ENV PYTHONPATH=$CODE_PATH
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV SAGEMAKER_PROGRAM train.py