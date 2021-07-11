# Image classification SageMaker training

This repository provide a template to use to train efficiently and effortlessly 
any image classification model. It is built to be quickly adaptable to new 
dataset, fully configurable and easy to deploy on pre configured GPU instance 
with SageMaker.

You will simply have to provide a folder containing images seprated in different 
sub-folders according to their class. Then you will need to configure the 
project. After that, using the project_manager python script, you will be able
to create 2 s3 buckets for your data and for your training outputs. Then you 
will be able to create a docker images which will be stored on ECR and then run 
on SageMaker as a training job. The resulting model will be stored on S3 with 
its performance metrics.

## Environemnt setup

First you need to setup a local virtual environement.

To do so use the following commande :

`python3 -m venv .env`

Then activate your local virtual environement with the following commande:

`. .env/bin/activate`

Finally install the local environement requirements with the following commande:

`pip install -r local_env_requirements.txt`


## Configure the application

The second thing you will need to do is to configure your application. For that 
you will need to edit the config file `container/modeling/config.py`. An example
is given for a classification job with flowers images.

## Split data in three sets

When configuring the applicatoin you should have, among other things, confugred
your local data directory. In order to split it in 3 directories you should use
the python script data_handler. Proceed the data split :

`python data_handler.py`
 
## Upload the dataset on S3

Once your data is splitted between the three dataset:
- training
- validation
- evaluation
You will then create the S3 buckets to store it, and upload the data on this 
bucket. You will do so by simply using the `project_manager` python script with 
the coresponding flag:

`python project_manager.py --init_s3_buckets`

(Prior to use that script you should configure your local AWS cli. To do so use
the commande `aws configure`.)

This will create two s3 buckets:
- `<config.PROJECT_NAME>.data`
- `<config.PROJECT_NAME>.output`

It will also upload the splited data from the local folder in the bucket 
`<config.PROJECT_NAME>.data`

It will also create the necessary S3 policy.

## Build the training docker image

In order to build the docker image for the training job based on the
configuration previously setup you simply have to run the folowing commande:

`python project_manager.py --build_and_push_docker_image`

It will also push your docker image on ECR.

## Launch the training job

The final step is to launch the training job. To do so simply use the commande:

`python project_manager.py --launch_training`

This will create the `training-job-config.json` file based on its template 
`training-job-config-template.json`. Then based on the created config file it
will launch the built docker image as a training job.

## Get the results of your training 

Once the training is finished, the best model will be saved on S3 in the bucket
`<config.PROJECT_NAME>.output`. It will be contained in the folder 
`<config.ALGORITHM_NAME>-YYYY-mm-dd-H-MM-SS/` so that you can run several 
trainings for the same project with different sets of hyperparamters and model 
architectures. The results will be contained in a zipped folder `model.tar.gz`.
Once unzipped il will have the folder architecture is the following:
- checkpoint
    | -assets
    | -keras_metadata.pb
    | -saved_model.pb
    | -variables
- classes_dict.pkl
- classification_report.pkl
- confusion_matrix.pkl
- model.h5


#### TODO:
- Check with jordan if the created buckets and ECR are well protected
- Add gradient decay
- add restart from checkpoint