import numpy as np
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam


def predictions_to_classes(predictions):
    return np.argmax(predictions, axis=-1)


def one_hot_to_sparse(tensor):
    return np.argmax(tensor, axis=1)


def unfreeze_model(model, optimizer=Adam(learning_rate=1e-5), metrics="acc"):
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=optimizer, loss=model.loss, metrics=metrics)
