from tensorflow.keras.applications import InceptionV3


# Application configs

DATA_NAME = "flowers"
PROJECT_NAME = "jac.test-sagemaker"
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

# Random seed set up to make repordiusable trainings
RANDOM_SEED = 42


