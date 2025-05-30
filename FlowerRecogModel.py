import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers
from keras.preprocessing.image import ImageDataGenerator, load_img 
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten


dirs = os.listdir('FlowerTrainingDataset/')

base_dir = 'FlowerTrainingDataset/'
img_size = 180
batch = 32

train_ds = tf.keras.utils.image_dataset_from_directory( base_dir,
                                                    seed = 123,
                                                    validation_split=0.2,
                                                    subset = 'training',
                                                    batch_size=batch,
                                                    image_size=(img_size,img_size))

has_elements = any(True for _ in train_ds.take(1))
if has_elements:
    print("Training dataset loaded")
else:
    print("Training dataset is empty")


val_ds = tf.keras.utils.image_dataset_from_directory( base_dir,
                                                    seed = 123,
                                                    validation_split=0.2,
                                                    subset = 'validation',
                                                    batch_size=batch,
                                                    image_size=(img_size,img_size))

has_elements = any(True for _ in val_ds.take(1))
if has_elements:
    print("Training dataset loaded")
else:
    print("Training dataset is empty")


flower_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

#--------------------------------------------------------------------------------------------------
# Date Augmentation:
#         Takes one image and apply random zoom, random flip and random rotation
#          Pros: Can use one image to make more images for sample

data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape = (img_size,img_size,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

#--------------------------------------------------------------------------------------------------
# Create Model:

def build_model():
    return Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5)
    ])

model = build_model()

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

model.summary()

history = model.fit(train_ds, epochs=15, validation_data=val_ds)
