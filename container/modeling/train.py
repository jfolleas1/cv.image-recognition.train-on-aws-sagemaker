import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, (3,3), activation='relu',
       input_shape=(*INPUT_IMG_SIZE, 3)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(output_layer_dim, activation='softmax')
  ])

  model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
  )

  return model


def train():


  train_generator, val_generator, eval_generator = build_data_pipelines()

  classes_dict = train_generator.class_indices

  print("classes_dict : \n ", classes_dict)

  model = build_model(output_layer_dim=len(classes_dict))
  
 
  print(f" $ ls {DATA_DIR} : ")
  print(os.listdir(DATA_DIR))
  print(f" $ ls {os.path.join(DATA_DIR, 'training')} : ")
  print(os.listdir(os.path.join(DATA_DIR, "training")))


  model.fit(train_generator, epochs=2, steps_per_epoch=20)
  test_loss, test_accuracy = model.evaluate(eval_generator, steps=100)
  print(f"Test accuracy: {test_accuracy}")

  model.save(OUTPUT_DIR + "model.h5")


if __name__=="__main__":
  train()