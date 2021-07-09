## Upload the dataset on S3

`aws s3 mb s3://YOUR-DATA-BUCKET-NAME`

aws s3 mb s3://jac.test-sagemaker.data

`aws s3 mb s3://YOUR-OUTPUT-BUCKET-NAME`

aws s3 mb s3://jac.test-sagemaker.output

`aws s3 sync DATA_FOLDER s3://YOUR-DATA-BUCKET-NAME`

aws s3 sync flowers s3://jac.test-sagemaker.data


```
account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
aws ecr get-login-password \
    --region ${region} \
| docker login \
    --username AWS \
    --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com
```

```
algorithm_name=sagemaker-tf-flower-example
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
docker build  -t ${algorithm_name} container
```

```
aws ecr create-repository --repository-name ${algorithm_name}
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
```


`aws iam create-role --role-name SagemakerRole --assume-role-policy-document file://./role-policy.json`

`aws iam attach-role-policy --role-name SagemakerRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess`

`aws iam put-role-policy --role-name SagemakerRole --policy-document file://./s3-policy.json --policy-name s3-policy`

```
aws sagemaker create-training-job --cli-input-json file://training-job-config.json
```


algorithm_name=sagemaker-tf-flower-example
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
docker build  -t ${algorithm_name} container
aws ecr create-repository --repository-name ${algorithm_name}
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}
aws sagemaker create-training-job --cli-input-json file://training-job-config.json


TODO:

Add gradient decay
add restart from checkpoint