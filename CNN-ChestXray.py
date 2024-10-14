import numpy as np
import pandas as pd

import os
import tensorflow as tf
import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, Flatten, BatchNormalization, Dense, MaxPooling2D,Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from tensorflow.keras import callbacks
import glob
main_path = "/kaggle/input/chest-xray/chest_xray"

IMG_SIZE = 224
BATCH = 32
SEED = 42
train_path = os.path.join(main_path,"train")
test_path=os.path.join(main_path,"test")

train_normal = glob.glob(train_path+"/NORMAL/*.jpeg")
train_pneumonia = glob.glob(train_path+"/PNEUMONIA/*.jpeg")

test_normal = glob.glob(test_path+"/NORMAL/*.jpeg")
test_pneumonia = glob.glob(test_path+"/PNEUMONIA/*.jpeg")
train_list = [x for x in train_normal]
train_list.extend([x for x in train_pneumonia])

df_train = pd.DataFrame(np.concatenate([['Normal']*len(train_normal) , ['Pneumonia']*len(train_pneumonia)]), columns = ['class'])
df_train['image'] = [x for x in train_list]

test_list = [x for x in test_normal]
test_list.extend([x for x in test_pneumonia])

df_test = pd.DataFrame(np.concatenate([['Normal']*len(test_normal) , ['Pneumonia']*len(test_pneumonia)]), columns = ['class'])
df_test['image'] = [x for x in test_list]
train_df, val_df = train_test_split(df_train, test_size = 0.20, random_state = SEED, stratify = df_train['class'])

train_df['class'] = train_df['class'].astype(str)
val_df['class'] = val_df['class'].astype(str)


train_datagen = ImageDataGenerator(rescale=1/255.,
                                   zoom_range=0.1,
                                   rotation_range=0.1,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)

val_datagen = ImageDataGenerator(rescale=1/255.)

ds_train = train_datagen.flow_from_dataframe(train_df,
                                              x_col='image',
                                              y_col='class',
                                              target_size=(IMG_SIZE, IMG_SIZE),
                                              class_mode='binary',  '
                                              batch_size=BATCH,
                                              seed=SEED)

ds_val = val_datagen.flow_from_dataframe(val_df,
                                          x_col='image',
                                          y_col='class',
                                          target_size=(IMG_SIZE, IMG_SIZE),
                                          class_mode='binary',
                                          batch_size=BATCH,
                                          seed=SEED)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-7,
    restore_best_weights=True,
)

plateau = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor = 0.2,
    patience = 2,
    min_delt = 1e-7,
    cooldown = 0,
    verbose = 1
)

model = tf.keras.models.Sequential()
model.add(layers.Input(shape = (224,224,3)))
model.add(layers.Conv2D(224, (3, 3), activation='relu'))
model.add(layers.AveragePooling2D((2,2)))
model.add(layers.Conv2D(448,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(448,(3,3), activation = 'relu'))
model.add(layers.AveragePooling2D((2,2)))
model.add(layers.Conv2D(448,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(448,(3,3), activation = 'relu'))
model.add(layers.AveragePooling2D((2,2)))
model.add(layers.Conv2D(448,(3,3), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(units = 448, activation = 'relu'))
model.add(layers.Dense(units = 224, activation = 'relu'))
model.add(layers.Dense(units = 78, activation = 'relu'))
model.add(layers.Dense(units = 26, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))
model.summary()
model.compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))
model.build()

model.fit(ds_train,
          epochs=50,
          validation_data=ds_val,
          callbacks=[early_stopping, plateau],
          steps_per_epoch=len(train_df) // BATCH,
          validation_steps=len(val_df) // BATCH)