from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import random
import tensorflow as tf
from typing import Tuple


# tran data has to be in train folder and validation data has to be in val folder
def get_train_val_generators(img_datagen: ImageDataGenerator,
                             data_dir: Path = Path('../data'),
                             color_mode: str = 'rgb',
                             batch_size: int = 128,
                             class_mode: str = 'categorical',
                             **kwargs):
    train_generator = img_datagen.flow_from_directory(os.path.join(data_dir, 'train'),
                                                      batch_size=batch_size,
                                                      color_mode=color_mode,
                                                      class_mode=class_mode,
                                                      **kwargs)
    validation_generator = img_datagen.flow_from_directory(os.path.join(data_dir, 'val'),
                                                           batch_size=batch_size,
                                                           color_mode=color_mode,
                                                           class_mode=class_mode,
                                                           **kwargs)
    return train_generator, validation_generator


def get_val_test_generators(img_datagen: ImageDataGenerator,
                            data_dir: Path = Path('../data'),
                            color_mode: str = 'rgb',
                            batch_size: int = 128,
                            class_mode: str = 'categorical',
                            **kwargs
                            ):
    validation_generator = img_datagen.flow_from_directory(os.path.join(data_dir, 'val'),
                                                           batch_size=batch_size,
                                                           color_mode=color_mode,
                                                           class_mode=class_mode,
                                                           **kwargs)

    test_generator = img_datagen.flow_from_directory(os.path.join(data_dir, 'test'),
                                                     batch_size=batch_size,
                                                     color_mode=color_mode,
                                                     class_mode=class_mode,
                                                     **kwargs)
    return validation_generator, test_generator


def numpy_memmap_generator(x_path: Path,
                           y_path: Path,
                           batch_size: int = 128,
                           shuffle_array: bool = True):
    while True:
        x = np.load(x_path, mmap_mode='r')
        y = np.load(y_path, mmap_mode='r')
        indexes = [i for i in range(x.shape[0])]

        if shuffle_array:
            random.shuffle(indexes)

        iterations = len(indexes) // batch_size + 1
        for i in range(iterations):
            slice_begin = i * batch_size
            slice_end = (i + 1) * batch_size
            slice_indexes = indexes[slice_begin: slice_end]
            yield x[slice_indexes], y[slice_indexes]


def get_train_val_image_datasets(train_path: Path,
                                 val_path: Path,
                                 image_size: Tuple[int, int],
                                 batch_size: int = 64,
                                 label_mode: str = 'categorical',
                                 cache_train: bool = False,
                                 cache_val: bool = True,
                                 train_mixup=False,
                                 mixup_alpha=0.2,
                                 **kwargs):

    train_ds1 = tf.keras.preprocessing.image_dataset_from_directory(train_path,
                                                                   label_mode=label_mode,
                                                                   batch_size=batch_size,
                                                                   image_size=image_size,
                                                                   **kwargs)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_path,
                                                                 label_mode=label_mode,
                                                                 batch_size=batch_size,
                                                                 image_size=image_size,
                                                                 shuffle=False,
                                                                 **kwargs)

    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE) if cache_val else val_ds.prefetch(tf.data.AUTOTUNE)

    if train_mixup:
        train_ds2 = tf.keras.preprocessing.image_dataset_from_directory(train_path,
                                                                        label_mode=label_mode,
                                                                        batch_size=batch_size,
                                                                        image_size=image_size,
                                                                        **kwargs)
        train_ds = tf.data.Dataset.zip((train_ds1, train_ds2))
        train_ds = train_ds.map(
            lambda ds_one, ds_two: __mix_up(ds_one, ds_two, alpha=mixup_alpha), num_parallel_calls=tf.data.AUTOTUNE
        )
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE) if cache_train else train_ds.prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = train_ds1.cache().prefetch(tf.data.AUTOTUNE) if cache_train else train_ds1.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def __mix_up(ds_one: tf.data.Dataset,
             ds_two: tf.data.Dataset,
             alpha: float):
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = __sample_beta_distribution(batch_size, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return images, labels


def __sample_beta_distribution(size: int, alpha: float):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=alpha)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=alpha)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)
