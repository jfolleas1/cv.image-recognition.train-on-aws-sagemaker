import os
import argparse

from datetime import datetime

import container.modeling.config as cfg

def build_s3_policy_from_template():
    """
    Build the AWS S3 policy based on the json file s3-policy-template.json
    """
    # Build the s3 policy file from the template
    with open('s3-policy-template.json', 'r') as f:
        s3_policy = f.read()
    project_name = cfg.PROJECT_NAME
    s3_policy = s3_policy.replace("[project_name]", project_name)
    # Save the built traing job config json file
    with open("s3-policy.json", "w") as f:
        f.write(s3_policy)

def init_s3_buckets():
    """
    Create the necessary AWS S3 bucket and the policies with it.
    """
    os.system(f"aws s3 mb s3://{cfg.PROJECT_NAME}.smtj.data")
    os.system(f"aws s3 mb s3://{cfg.PROJECT_NAME}.smtj.output")
    os.system(f"aws s3 sync {cfg.DATA_LOCAL_DIR} s3://{cfg.PROJECT_NAME}" +\
        f".smtj.data/{cfg.DATA_NAME}")
    os.system(f"aws iam create-role --role-name {cfg.PROJECT}-SagemakerRole " +\
        "--assume-role-policy-document file://./role-policy.json")
    os.system(f"aws iam attach-role-policy --role-name {cfg.PROJECT}-SagemakerRole " +\
        "--policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess")
    build_s3_policy_from_template()
    os.system(f"aws iam put-role-policy --role-name {cfg.PROJECT}-SagemakerRole " +\
        "--policy-document file://./s3-policy.json --policy-name s3-policy")

def get_aws_account():
    """
    Get the AWS account ot use for the training deployement
    """
    stream = os.popen('aws sts get-caller-identity --query Account --output text')
    account = stream.read()[:-1]
    if account is None:
        print("AWS account not found, please configure AWS cli with " +\
            "`aws configure` command")
        raise RuntimeError('AWS account not found')
    return account

def get_aws_region():
    """
    Get the AWS region ot use for the training deployement
    """
    # stream = os.popen('aws configure get region')
    # region = stream.read()[:-1]
    # if region is None:
    #     print("AWS region not found, please configure AWS cli with " +\
    #         "`aws configure` command")
    #     raise RuntimeError('AWS region not found')
    region = cfg.PROJECT_REGION
    return region

def get_fullname():
    """
    Get the full name of the docker images to use on AWS ECR.
    """
    account = get_aws_account()
    region = get_aws_region()
    algorithm_name=cfg.ALGORITHM_NAME
    fullname=f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
    return fullname

def build_and_push_docker_image():
    """
    Build and push the docker image which will be used for the training job.
    """
    # login on docker with AWS credentials
    account = get_aws_account()
    region = get_aws_region()
    os.system(f"aws ecr get-login-password " +\
              f"--region {region} " +\
              f"| docker login " +\
              f"--username AWS " +\
              f"--password-stdin {account}.dkr.ecr.{region}.amazonaws.com")
    # Build the training docker image
    algorithm_name = cfg.ALGORITHM_NAME
    os.system(f"docker build  -t {algorithm_name} container")
    os.system(f"aws ecr create-repository --repository-name {algorithm_name}")
    os.system(f"aws ecr batch-delete-image --repository-name {algorithm_name}" +\
            f" --image-ids imageTag=latest")
    fullname = get_fullname()
    os.system(f"docker tag {algorithm_name} {fullname}")
    os.system(f"docker push {fullname}")

def launch_training():
    """
    Launch the training job.
    """
    # Build the config file for the training job
    curr_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    region = get_aws_region()
    account = get_aws_account()
    algorithm_name = cfg.ALGORITHM_NAME
    project_name = cfg.PROJECT_NAME
    instance_type = cfg.INSTANCE_TYPE
    data_name = cfg.DATA_NAME
    customer = cfg.CUSTOMER
    project_sub_name = cfg.PROJECT
    with open('training-job-config-template.json', 'r') as f:
        training_job_config = f.read()

    training_job_config = training_job_config.replace("[curr_datetime]",\
                                                      curr_datetime)
    training_job_config = training_job_config.replace("[region]",\
                                                      region)
    training_job_config = training_job_config.replace("[account]",\
                                                      account)
    training_job_config = training_job_config.replace("[algorithm_name]",\
                                                      algorithm_name)
    training_job_config = training_job_config.replace("[project_name]",\
                                                      project_name)
    training_job_config = training_job_config.replace("[instance_type]",\
                                                      instance_type)
    training_job_config = training_job_config.replace("[data_name]",\
                                                      data_name)
    training_job_config = training_job_config.replace("[customer]",\
                                                      customer)
    training_job_config = training_job_config.replace("[project_sub_name]",\
                                                      project_sub_name)

                                                      
    # Save the built traing job config json file
    with open("training-job-config.json", "w") as f:
        f.write(training_job_config)
    # Launch the training job
    os.system("aws sagemaker create-training-job --cli-input-json " +\
        "file://training-job-config.json")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_s3_buckets",
        help="Use flag if you want to init the s3 buckets",
        action='store_true')

    parser.add_argument("--build_and_push_docker_image",
        help="Use the flag if you want to build and push the training docker" +\
             " image",
        action='store_true')

    parser.add_argument("--launch_training",
        help="Use the flag if you want to launch your model training",
        action='store_true')

    args = parser.parse_args() 


    if args.init_s3_buckets:
        init_s3_buckets()

    if args.build_and_push_docker_image:
        build_and_push_docker_image()

    if args.launch_training:
        launch_training()
