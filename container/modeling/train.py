import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import warnings

import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import config as cfg
from utils import get_nb_step

# location of the data dir and output dir in the container
DATA_DIR = os.path.join("/opt/ml/input/data", cfg.DATA_NAME)
OUTPUT_DIR = "/opt/ml/model/"

def build_data_pipelines():
  """
  Build the three data generator for training, validaton and evaluation.
  The Training generator also include some data augmentation based on the 
  cofiguration.

  Returns:
    - train_generator (ImageDataGenerator): The data generator for the training
      set.
    - val_generator (ImageDataGenerator): The data generator for the validation
      set.
    - eval_generator (ImageDataGenerator): The data generator for the evaluation
      set.

  """
  # Geting the local path for the three set directories
  train_data_path = os.path.join(DATA_DIR, 'training')
  val_data_path = os.path.join(DATA_DIR, 'validation')
  eval_data_path = os.path.join(DATA_DIR, 'evaluation')
  # Build the class liste based on the directories names to be sure that the 
  # different generators have the same classes orders.
  classes = sorted(os.listdir(train_data_path))
  # Setup training data generator with data augmentation
  train_augmentor = ImageDataGenerator(
    rescale = cfg.PIXEL_RESCALE,
    rotation_range=cfg.TRAINING_ROTATION_RANGE,
    zoom_range=cfg.TRAINING_ZOOM_RANGE,
    width_shift_range=cfg.TRAINING_WIDTH_SHIFT_RANGE,
    height_shift_range=cfg.TRAINING_HEIGTH_SHIFT_RANGE,
    shear_range=cfg.TRAINING_SHEAR_RANGE,
    horizontal_flip=cfg.TRAINING_HORIZONTAL_FLIP,
    fill_mode="nearest"
  )
  # Setup validation data generator
  val_augmentor = ImageDataGenerator(
    rescale = cfg.PIXEL_RESCALE
  )
  # Setup evaluation data generator
  eval_augmentor = ImageDataGenerator(
    rescale = cfg.PIXEL_RESCALE
  )
  # Build the data generator from the local training data directory
  train_generator = train_augmentor.flow_from_directory(
    train_data_path,
    classes=classes,
    class_mode="categorical",
    target_size=cfg.INPUT_IMG_SIZE,
    color_mode="rgb",
    shuffle=True,
    batch_size=cfg.BATCH_SIZE
  )
  # Build the data generator from the local validatoin data directory
  val_generator = val_augmentor.flow_from_directory(
    val_data_path,
    classes=classes,
    class_mode="categorical",
    target_size=cfg.INPUT_IMG_SIZE,
    color_mode="rgb",
    shuffle=False,
    batch_size=cfg.BATCH_SIZE
  )
  # Build the data generator from the local evaluation data directory
  eval_generator = eval_augmentor.flow_from_directory(
    eval_data_path,
    classes=classes,
    class_mode="categorical",
    target_size=cfg.INPUT_IMG_SIZE,
    color_mode="rgb",
    shuffle=False,
    batch_size=cfg.BATCH_SIZE
  )
  # Return the data generators
  return train_generator, val_generator, eval_generator

def build_model(output_layer_dim):
  """
  Buidl the model which will be train and compile it based on the configuration
  indicated by the user.

  Params:
    - output_layer_dim (int): The dimension of the output layer. Must be the 
      number of class that the model must make its predictions on.

  Returns:
    - model (tensorflow.keras.model): The model which is ready to be trained.
  """
  # Load the base model
  base_model = cfg.MODEL_BASE(weights=cfg.MODEL_WEIGHTS, include_top=False,\
      input_tensor=tf.keras.layers.Input(shape=(*cfg.INPUT_IMG_SIZE, 3)))
  # Build the head of the model 
  head_model = base_model.output
  head_model = tf.keras.layers.Flatten()(head_model)
  for head_dim, head_dropout in cfg.HEAD_DIMENSION_DROPOUTS:
    head_model = tf.keras.layers.Dense(head_dim, activation='relu')(head_model)
    head_model = tf.keras.layers.Dropout(head_dropout)(head_model)
  head_model = tf.keras.layers.Dense(output_layer_dim,\
      activation="softmax")(head_model)
  # Merge the two parts to create the whole model
  model = tf.keras.models.Model(inputs=base_model.input, outputs=head_model)
  # Fix the weight of the base model
  for layer in base_model.layers:
    layer.trainable = False
  # Compile the model
  model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=cfg.MODEL_LEARNING_RATE),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
  )
  # Return the ready to train model
  return model

def build_callbacks():
  """
  Buidl the callback which will help to save the best model from the training 
  and avoid overfiting with an early stopping

  Returns:
    - early_stopping (EarlyStopping): The early stopping of the training build
      accordingly to the user configuration.
    - ckpt_saver (ModelCheckpoint): The model checkpoint which will help to keep
      only the best model from the training.
  """
  # Build early stopping
  early_stopping = EarlyStopping(monitor=cfg.EARLY_STOPPING_MONITOR,
                                 patience=cfg.EARLY_STOPPING_PATIENCE)
  # Build model checkpoints
  ckpt_saver = ModelCheckpoint(
      os.path.join(OUTPUT_DIR, "checkpoints"),
      monitor=cfg.CHECKPOINT_MONITOR,
      mode=cfg.CHECKPOINT_MODE,
      save_best_only=cfg.CHECKPOINT_SAVE_BEST_ONLY,
      save_freq=cfg.CHECKPOINT_SAVE_FREQ,
      save_weights_only=False,
      verbose=1
  )
  # return callbacks
  return early_stopping, ckpt_saver


def train():
  """
  Set up the data generators to train the model. Buidl the mdoel accordingly 
  to the configuration and train the built model on the given images.
  It will also evaluate the best model perfomance and save it in a local
  directory which will be pushed on S3.
  """
  # Build the data generators
  train_generator, val_generator, eval_generator = build_data_pipelines()
  # Get the classes mapping dictionnairy from the data generators
  classes_dict = train_generator.class_indices
  # Print the classes mapping
  print("classes_dict : \n ", classes_dict)
  # Build the model to train
  model = build_model(output_layer_dim=len(classes_dict))
  # Prepare the callbacks
  early_stopping, ckpt_saver = build_callbacks()
  # Train the mdoel on the training data generator indicating the validation 
  # data generator to use at the end on each epochs.
  model.fit(
    train_generator,
    steps_per_epoch=get_nb_step(DATA_DIR, 'training'),
    validation_data=val_generator,
    validation_steps=get_nb_step(DATA_DIR, 'validation'),
    epochs=cfg.NB_EPOCHS,
    callbacks=[early_stopping, ckpt_saver]
  )
  # Once the training is finished:
  # Load best model
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.load_weights(os.path.join(OUTPUT_DIR, 'checkpoints'))
  # Evaluate the model
  print("[INFO] Evaluation phase...")
  # Make the predictions on the test images from the evaluation dataset
  predictions = model.predict_generator(eval_generator)
  predictions_idxs = np.argmax(predictions, axis=1)
  # Build a classification report
  my_classification_report = classification_report(eval_generator.classes,
      predictions_idxs, target_names=eval_generator.class_indices.keys())
  # Build a confusion matrix
  my_confusion_matrix = confusion_matrix(eval_generator.classes,
      predictions_idxs)
  # Print the classifiyer performances
  print("[INFO] Classification report : ")
  print(my_classification_report)

  print("[INFO] Confusion matrix : ")
  print(my_confusion_matrix)

  # Save the best model and it's results and config
  model.save(os.path.join(OUTPUT_DIR, "model.h5"))
  pickle.dump(my_classification_report, open(os.path.join(OUTPUT_DIR,
      "classification_report.pkl"), "wb" ))
  pickle.dump(my_confusion_matrix, open(os.path.join(OUTPUT_DIR,
      "confusion_matrix.pkl"), "wb" ))
  pickle.dump(classes_dict, open(os.path.join(OUTPUT_DIR,
      "classes_dict.pkl"), "wb" ))
  config_dict = cfg.get_config_dict() 
  pickle.dump(config_dict, open(os.path.join(OUTPUT_DIR,
      "config_dict.pkl"), "wb" ))

if __name__=="__main__":
  # Set the random seed to be able to reproduce a training 
  tf.random.set_seed(cfg.RANDOM_SEED)
  train()