# simplified keras
Common used actions in keras

## Table of contents
* [General info](#general-info)
* [Libraries](#libraries)
* [Setup](#setup)
* [Documentation](#documentation)
* [PyPi](#pypi)
* [TODO](#todo)
* [Development](#development)
* [Status](#status)
* [Contact](#contact)

## General info
This package is a set of common used actions in keras. At this moment includes:
* [Fast train and validation generators creation](#generators)
* [Default callbacks](#default-callbacks)
* [Accuracy and Loss plot](#accuracy-and-loss-plot)
* [Predictions with image plot](#predictions-with-image-plot)
* [Histogram with CDF and image plot](#histogram-with-cdf-and-image-plot)

## Libraries
- Keras - version 2.4.3
- Matplotlib - version 3.3.3
- NumPy - version 1.19.4

## Setup
* Install from PyPi: `pip install simplified-keras`

## Documentation

### Generators
```python
from keras.preprocessing.image import ImageDataGenerator
from simplified_keras.dir_flow_generators import get_train_val_generators

img_size = (48, 48)
img_datagen = ImageDataGenerator(rescale=1/255)

train_generator, validation_generator = get_train_val_generators(img_datagen, data_dir='../data/normal/',
                                                                 color_mode='grayscale', target_size=img_size)
```
Signature:

```python
def get_train_val_generators(img_datagen: ImageDataGenerator, data_dir='../data/', target_size=None, 
                             color_mode='rgb', batch_size=128, class_mode='categorical')
```

### Default callbacks

```python
from simplified_keras.default_callbacks import get_default_callbacks

callbacks = get_default_callbacks('vgg16_calssifier')

hist = model.fit(train_generator, steps_per_epoch=train_steps, validation_data=validation_generator, 
                 validation_steps=valid_steps, epochs=100, callbacks=callbacks, verbose=2)
```
Signature:

```python
def get_default_callbacks(model_name):
    return [
        clb.ReduceLROnPlateau(monitor='val_acc', factor=0.5, min_lr=1e-6, patience=3, verbose=1),
        clb.EarlyStopping(monitor='val_acc', patience=7, verbose=1),
        clb.ModelCheckpoint(monitor='val_acc', filepath=f'../models/{model_name}.h5',
                            save_best_only=True, verbose=1)
    ]
```

### Accuracy and Loss plot

```python
from simplified_keras.plots.history_plots import plot_acc_and_loss

history = model.fit(train_gen, teps_per_epoch=train_steps, epochs=5, alidation_data=val_gen, 
                    validation_steps=val_steps, callbacks=callbacks)

plot_acc_and_loss(history)
```

Result:

[![history.png](https://i.postimg.cc/YqYgStNM/t.png)](https://postimg.cc/8jksKQn0)

### Predictions with image plot

```python
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from simplified_keras.dir_flow_generators import get_train_val_generators
from simplified_keras.plots import plot_predictions_with_img

img_size = (48, 48)
img_datagen = ImageDataGenerator(rescale=1/255)

_, validation_generator = get_train_val_generators(img_datagen, data_dir='../data/normal/',
                                                   color_mode='grayscale', target_size=img_size)
model = load_model('../models/standard_model.h5')

batch, labels = validation_generator.next()
preds = model.predict(batch)

named_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
plot_predictions_with_img(1, preds, labels, batch, named_labels)
```

Result:

[![p.png](https://i.postimg.cc/Hs2dD4Tw/p.png)](https://postimg.cc/ykkw0Rgx)

### Histogram with CDF and image plot

```python
import cv2
from simplified_keras.plots import plot_gray_img_with_histogram

img = cv2.imread(f'{src_train_path}/0/241.png', 0)
plot_gray_img_with_histogram(img)
img2 = stretch_histogram(img)
plot_gray_img_with_histogram(img2)
```

Result:

[![history1.png](https://i.postimg.cc/JzwN3Pc3/h1.png)](https://i.postimg.cc/JzwN3Pc3/h1.png)
[![history2.png](https://i.postimg.cc/x1KK6Mtw/h2.png)](https://i.postimg.cc/x1KK6Mtw/h2.png)


## PyPi
[simplified-keras](add-link)

## TODO
- nothing :)

## Development
Want to contribute? Great!

To fix a bug or enhance an existing module, follow these steps:

* Fork the repo
* Create a new branch (`git checkout -b improve-feature`)
* Make the appropriate changes in the files
* **Verify if they are correct**
* Add changes to reflect the changes made
* Commit changes
* Push to the branch (`git push origin improve-feature`)
* Create a Pull Request

## Status
Library is: _in progress_

## Contact
albert.lis.1996@gmail.com - feel free to contact me!
