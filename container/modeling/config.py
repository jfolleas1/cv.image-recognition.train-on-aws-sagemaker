from tensorflow.keras.applications import InceptionV3


# Application configs

DATA_NAME = "flowers"
CUSTOMER = "jac"
PROJECT = "test-sagemaker"
PROJECT_NAME = CUSTOMER + "." + PROJECT # Do not modify
PROJECT_REGION = "eu-west-1"
ALGORITHM_NAME = "sagemaker-tf-flower-example"

# Local data config

DATA_LOCAL_DIR = "./flowers"
PERCENTAGE_VAL = 0.1
PERCENTAGE_EVAL = 0.1

# Model config

MODEL_BASE = InceptionV3
MODEL_WEIGHTS = "imagenet"
HEAD_DIMENSION_DROPOUTS = [(512, 0.5)]
MODEL_LEARNING_RATE = 1e-5

# Training cofigs

INPUT_IMG_SIZE = (150, 150)
BATCH_SIZE = 128

## Parameters of tensorflow.keras.preprocessing.image.ImageDataGenerator
PIXEL_RESCALE = 1. / 255
TRAINING_ROTATION_RANGE = 25 # default: 0
TRAINING_ZOOM_RANGE = 0.15 # default: 0
TRAINING_WIDTH_SHIFT_RANGE = 0.2 # default: 0
TRAINING_HEIGTH_SHIFT_RANGE = 0.2 # default: 0
TRAINING_SHEAR_RANGE = 0.15 # default: 0
TRAINING_HORIZONTAL_FLIP = True # default: False
TRAINING_FILL_MODE = "nearest"

## Callbacks

EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 5
CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'
# Advised to not modify CHECKPOINT_SAVE_BEST_ONLY without changing inner code
CHECKPOINT_SAVE_BEST_ONLY = True 
CHECKPOINT_SAVE_FREQ = 'epoch'

## fit
STEP_PER_EPOCH = 20 # default: 'auto'
VALIDATION_STEPS = 20 # default: 'auto'
EVALUATION_STEPS = 20 # default: 'auto'
NB_EPOCHS = 2

# Others

# Random seed set up to make trainings reproductable
RANDOM_SEED = 42
# Instance on which the training is runing
INSTANCE_TYPE = "ml.g4dn.xlarge"


#############################################
############# DO NOT EDIT BELOW #############
#############################################

def get_config_dict():
    """
    Buidl a dictionnary containing all the config variables and return it.
    """
    config_dict = {
        "DATA_NAME": DATA_NAME,
        "PROJECT_NAME": PROJECT_NAME,
        "PROJECT_REGION": PROJECT_REGION,
        "ALGORITHM_NAME": ALGORITHM_NAME,
        "DATA_LOCAL_DIR": DATA_LOCAL_DIR,
        "PERCENTAGE_VAL": PERCENTAGE_VAL,
        "PERCENTAGE_EVAL": PERCENTAGE_EVAL,
        "MODEL_BASE": MODEL_BASE,
        "MODEL_WEIGHTS": MODEL_WEIGHTS,
        "HEAD_DIMENSION_DROPOUTS": HEAD_DIMENSION_DROPOUTS,
        "MODEL_LEARNING_RATE": MODEL_LEARNING_RATE,
        "INPUT_IMG_SIZE": INPUT_IMG_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "PIXEL_RESCALE": PIXEL_RESCALE,
        "TRAINING_ROTATION_RANGE": TRAINING_ROTATION_RANGE,
        "TRAINING_ZOOM_RANGE": TRAINING_ZOOM_RANGE,
        "TRAINING_WIDTH_SHIFT_RANGE": TRAINING_WIDTH_SHIFT_RANGE,
        "TRAINING_HEIGTH_SHIFT_RANGE": TRAINING_HEIGTH_SHIFT_RANGE,
        "TRAINING_SHEAR_RANGE": TRAINING_SHEAR_RANGE,
        "TRAINING_HORIZONTAL_FLIP": TRAINING_HORIZONTAL_FLIP,
        "TRAINING_FILL_MODE": TRAINING_FILL_MODE,
        "EARLY_STOPPING_MONITOR": EARLY_STOPPING_MONITOR,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "CHECKPOINT_MONITOR": CHECKPOINT_MONITOR,
        "CHECKPOINT_MODE": CHECKPOINT_MODE,
        "CHECKPOINT_SAVE_BEST_ONLY": CHECKPOINT_SAVE_BEST_ONLY,
        "CHECKPOINT_SAVE_FREQ": CHECKPOINT_SAVE_FREQ,
        "STEP_PER_EPOCH": STEP_PER_EPOCH,
        "VALIDATION_STEPS": VALIDATION_STEPS,
        "EVALUATION_STEPS": EVALUATION_STEPS,
        "NB_EPOCHS": NB_EPOCHS,
        "RANDOM_SEED": RANDOM_SEED,
        "INSTANCE_TYPE": INSTANCE_TYPE
    }
    return config_dict