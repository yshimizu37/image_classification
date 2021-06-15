import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def image_classification(dataset_dir, model_dir, batch_size, epochs, IMG_HEIGHT=150, IMG_WIDTH=150):
  train_dir = os.path.join(dataset_dir, 'train')
  validation_dir = os.path.join(dataset_dir, 'validation')

  train_cats_dir = os.path.join(train_dir, 'cats')  # 学習用の猫画像のディレクトリ
  train_dogs_dir = os.path.join(train_dir, 'dogs')  # 学習用の犬画像のディレクトリ
  validation_cats_dir = os.path.join(validation_dir, 'cats')  # 検証用の猫画像のディレクトリ
  validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # 検証用の犬画像のディレクトリ


  num_cats_tr = len(os.listdir(train_cats_dir))
  num_dogs_tr = len(os.listdir(train_dogs_dir))

  num_cats_val = len(os.listdir(validation_cats_dir))
  num_dogs_val = len(os.listdir(validation_dogs_dir))

  total_train = num_cats_tr + num_dogs_tr
  total_val = num_cats_val + num_dogs_val

  image_gen_train = ImageDataGenerator(
                      rescale=1./255,
                      rotation_range=45,
                      width_shift_range=.15,
                      height_shift_range=.15,
                      horizontal_flip=True,
                      zoom_range=0.5
                      )

  train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                      directory=train_dir,
                                                      shuffle=True,
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      class_mode='binary')

  """### 検証データジェネレータの構築

  一般に、データ拡張は学習サンプルのみに適用します。今回は、 `ImageDataGenerator` を使用して検証画像に対してリスケールのみを実施し、バッチに変換します。
  """

  image_gen_val = ImageDataGenerator(rescale=1./255)

  val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                  directory=validation_dir,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  class_mode='binary')

  model = Sequential([
      Conv2D(16, 3, padding='same', activation='relu', 
            input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
      MaxPooling2D(),
      Dropout(0.2),
      Conv2D(32, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Conv2D(64, 3, padding='same', activation='relu'),
      MaxPooling2D(),
      Dropout(0.2),
      Flatten(),
      Dense(512, activation='relu'),
      Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.summary()

  history = model.fit_generator(
      train_data_gen,
      steps_per_epoch=total_train // batch_size,
      epochs=epochs,
      validation_data=val_data_gen,
      validation_steps=total_val // batch_size
  )
  hist_path = os.path.join(model_dir, "history.csv")
  hist_df = pd.DataFrame(history.history)
  hist_df.to_csv(hist_path)

def main() -> None:
    parser = argparse.ArgumentParser(description='説明')

    parser.add_argument('-d', '--datadir', default="/mnt/dataset", help='dataset directory')
    parser.add_argument('-m', '--modeldir', default="/mnt/model", help='model directory')
    parser.add_argument('-b', '--batchsize', default=128, help='batch size')
    parser.add_argument('-e', '--epochs', default=15, help='epochs')

    args = parser.parse_args()

    image_classification(dataset_dir=args.datadir, model_dir=args.modeldir, batch_size=int(args.batchsize), epochs=int(args.epochs))

if __name__ == "__main__":
    main()