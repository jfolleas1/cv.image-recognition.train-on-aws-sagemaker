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

DATA_DIR = os.path.join("/opt/ml/input/data", cfg.DATA_NAME)
# DATA_DIR = "./flowers"
OUTPUT_DIR = "/opt/ml/model/"



def build_data_pipelines():
  train_data_path = os.path.join(DATA_DIR, 'training')
  val_data_path = os.path.join(DATA_DIR, 'validation')
  eval_data_path = os.path.join(DATA_DIR, 'evaluation')

  train_augmentor = ImageDataGenerator(
    rescale = cfg.PIXEL_RESCALE,
    rotation_range=cfg.TRAINING_ROTATION_RANGE,
    zoom_range=cfg.TRAINING_ZOOM_RANGE,
    width_shift_range=cfg.TRAINING_WIDTH_SHIFT_RANGE,
    height_shift_range=cfg.TRAINING_HEIGTH_SHIFT_RANGE,
    shear_rang=cfg.TRAINING_SHEAR_RANGE,
    horizontal_flip=cfg.TRAINING_HORIZONTAL_FLIP,
    fill_mode="nearest"
  )

  val_augmentor = ImageDataGenerator(
    rescale = cfg.PIXEL_RESCALE
  )

  eval_augmentor = ImageDataGenerator(
    rescale = cfg.PIXEL_RESCALE
  )

  train_generator = train_augmentor.flow_from_directory(
    train_data_path,
    class_mode="categorical",
    target_size=cfg.INPUT_IMG_SIZE,
    color_mode="rgb",
    shuffle=True,
    batch_size=cfg.BATCH_SIZE
  )

  val_generator = val_augmentor.flow_from_directory(
    val_data_path,
    class_mode="categorical",
    target_size=cfg.INPUT_IMG_SIZE,
    color_mode="rgb",
    shuffle=False,
    batch_size=cfg.BATCH_SIZE
  )

  eval_generator = eval_augmentor.flow_from_directory(
    eval_data_path,
    class_mode="categorical",
    target_size=cfg.INPUT_IMG_SIZE,
    color_mode="rgb",
    shuffle=False,
    batch_size=cfg.BATCH_SIZE
  )

  return train_generator, val_generator, eval_generator

def build_model(output_layer_dim):
  # Load the base model
  base_model = cfg.MODEL_BASE(weights=cfg.MODEL_WEIGHTS, include_top=False,\
      input_tensor=tf.keras.layers.Input(shape=(*cfg.INPUT_IMG_SIZE, 3)))

  # Build the head
  head_model = base_model.output
  head_model = tf.keras.layers.Flatten()(head_model)
  for head_dim, head_dropout in cfg.HEAD_DIMENSION_DROPOUTS:
    head_model = tf.keras.layers.Dense(head_dim, activation='relu')(head_model)
    head_model = tf.keras.layers.Dropout(head_dropout)(head_model)
  head_model = tf.keras.layers.Dense(output_layer_dim,\
      activation="softmax")(head_model)

  # Build the whole model
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

  return model

def build_callbacks():
  early_stopping = EarlyStopping(monitor=cfg.EARLY_STOPPING_MONITOR,
                                 patience=cfg.EARLY_STOPPING_PATIENCE)

  ckpt_saver = ModelCheckpoint(
      os.path.join(cfg.OUTPUT_DIR, "checkpoints"),
      monitor=cfg.CHECKPOINT_MONITOR,
      mode=cfg.CHECKPOINT_MODE,
      save_best_only=cfg.CHECKPOINT_SAVE_BEST_ONLY,
      save_freq=cfg.CHECKPOINT_SAVE_FREQ,
      save_weights_only=False,
      verbose=1
  )

  return early_stopping, ckpt_saver


def train():
  train_generator, val_generator, eval_generator = build_data_pipelines()

  classes_dict = train_generator.class_indices

  print("classes_dict : \n ", classes_dict)

  model = build_model(output_layer_dim=len(classes_dict))

  early_stopping, ckpt_saver = build_callbacks()

  model.fit(
    train_generator,
    steps_per_epoch=get_nb_step(DATA_DIR, 'training'),
    validation_data=val_generator,
    validation_steps=get_nb_step(DATA_DIR, 'validation'),
    epochs=cfg.NB_EPOCHS,
    callbacks=[early_stopping, ckpt_saver]
  )

  print('ls output dir : ')
  print(os.listdir(cfg.OUTPUT_DIR))
  print('ls output/checkpoints dir : ')
  print(os.listdir(os.path.join(cfg.OUTPUT_DIR, 'checkpoints')))

  # Load best model
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.load_weights(os.path.join(cfg.OUTPUT_DIR, 'checkpoints'))
  
  print("[INFO] Evaluation phase...")

  predictions = model.predict_generator(eval_generator)
  predictions_idxs = np.argmax(predictions, axis=1)

  my_classification_report = classification_report(eval_generator.classes,
      predictions_idxs, target_names=eval_generator.class_indices.keys())

  my_confusion_matrix = confusion_matrix(eval_generator.classes,
      predictions_idxs)
  
  # Print the classifiyer performances
  
  print("[INFO] Classification report : ")
  print(my_classification_report)

  print("[INFO] Confusion matrix : ")
  print(my_confusion_matrix)

  # Save the best model and it's results

  model.save(os.path.join(cfg.OUTPUT_DIR, "model.h5"))
  pickle.dump(my_classification_report, open(os.path.join(cfg.OUTPUT_DIR,
      "classification_report.pkl"), "wb" ))
  pickle.dump(my_confusion_matrix, open(os.path.join(cfg.OUTPUT_DIR,
      "confusion_matrix.pkl"), "wb" ))

if __name__=="__main__":
  tf.random.set_seed(RANDOM_SEED)

  train()