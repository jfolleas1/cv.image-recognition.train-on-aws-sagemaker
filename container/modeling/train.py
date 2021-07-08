import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import warnings

import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from config import INPUT_IMG_SIZE, BATCH_SIZE

DATA_DIR = "/opt/ml/input/data/flowers"
# DATA_DIR = "./flowers"
OUTPUT_DIR = "/opt/ml/model/"



def build_data_pipelines():
  train_data_path = os.path.join(DATA_DIR, 'training')
  val_data_path = os.path.join(DATA_DIR, 'validation')
  eval_data_path = os.path.join(DATA_DIR, 'evaluation')

  train_augmentor = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range=25,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
  )

  val_augmentor = ImageDataGenerator(
    rescale = 1. / 255
  )

  eval_augmentor = ImageDataGenerator(
    rescale = 1. / 255
  )

  train_generator = train_augmentor.flow_from_directory(
    train_data_path,
    class_mode="categorical",
    target_size=INPUT_IMG_SIZE,
    color_mode="rgb",
    shuffle=True,
    batch_size=BATCH_SIZE
  )

  val_generator = val_augmentor.flow_from_directory(
    val_data_path,
    class_mode="categorical",
    target_size=INPUT_IMG_SIZE,
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE
  )

  eval_generator = eval_augmentor.flow_from_directory(
    eval_data_path,
    class_mode="categorical",
    target_size=INPUT_IMG_SIZE,
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE
  )

  return train_generator, val_generator, eval_generator

def build_model(output_layer_dim):

  base_model = InceptionV3(weights="imagenet", include_top=False,\
      input_tensor=tf.keras.layers.Input(shape=(*INPUT_IMG_SIZE, 3)))

  head_model = base_model.output
  head_model = tf.keras.layers.Flatten()(head_model)
  head_model = tf.keras.layers.Dense(512, activation='relu')(head_model)
  head_model = tf.keras.layers.Dropout(0.5)(head_model)
  head_model = tf.keras.layers.Dense(output_layer_dim,\
      activation="softmax")(head_model)

  model = tf.keras.models.Model(inputs=base_model.input, outputs=head_model)

  for layer in base_model.layers:
    layer.trainable = False

  model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
  )

  return model

def build_callbacks():
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)

  # path_to_save_model_base = CHECKPOINT_FILEPATH.rsplit('/', 1)[0]
  # if not os.path.isdir(path_to_save_model_base):
  # os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"))

  ckpt_saver = ModelCheckpoint(
      os.path.join(OUTPUT_DIR, "checkpoints"),
      monitor="val_accuracy",
      mode='max',
      save_best_only=True,
      save_freq='epoch',
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

  print("NEW IMAGE : ")

  model.fit(
    train_generator,
    steps_per_epoch=20,#total_train_imgs // batch_size,
    validation_data=val_generator,
    validation_steps=20, #total_val_imgs // batch_size,
    epochs=2,
    callbacks=[early_stopping, ckpt_saver]
  )

  print('ls output dir : ')
  print(os.listdir(OUTPUT_DIR))
  print('ls output/checkpoints dir : ')
  print(os.listdir(os.path.join(OUTPUT_DIR, 'checkpoints')))

  # Load best model
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.load_weights(os.path.join(OUTPUT_DIR, 'checkpoints'))
  
  print("[INFO] Evaluation phase...")

  predictions = model.predict_generator(eval_generator)
  predictions_idxs = np.argmax(predictions, axis=1)

  my_classification_report = classification_report(eval_generator.classes,
      predictions_idxs, target_names=eval_generator.class_indices.keys())

  my_confusion_matrix = confusion_matrix(eval_generator.classes,
      predictions_idxs)

  print("[INFO] Classification report : ")
  print(my_classification_report)

  print("[INFO] Confusion matrix : ")
  print(my_confusion_matrix)

  # Save the best model and it's results

  model.save(os.path.join(OUTPUT_DIR, "model.h5"))
  pickle.dump(my_classification_report, open(os.path.join(OUTPUT_DIR,
      "classification_report.pkl"), "wb" ))
  pickle.dump(my_confusion_matrix, open(os.path.join(OUTPUT_DIR,
      "confusion_matrix.pkl"), "wb" ))

if __name__=="__main__":
  tf.random.set_seed(42)

  train()