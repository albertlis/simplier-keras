from keras.preprocessing.image import ImageDataGenerator


# tran data has to be in train folder and validation data has to be in val folder
def get_train_val_generators(img_datagen: ImageDataGenerator, data_dir='../data/', target_size=None, color_mode='rgb',
                             batch_size=128, class_mode='categorical'):
    train_generator = img_datagen.flow_from_directory(f'{data_dir}train/',
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      color_mode=color_mode,
                                                      class_mode=class_mode)
    validation_generator = img_datagen.flow_from_directory(f'{data_dir}val/',
                                                           target_size=target_size,
                                                           batch_size=batch_size,
                                                           color_mode=color_mode,
                                                           class_mode=class_mode)
    return train_generator, validation_generator
