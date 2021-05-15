import io
from pathlib import Path
from typing import Callable, Tuple, List
import numpy as np
from simplified_keras.metrics import get_confusion_matrixes
from simplified_keras.plots import plot_confusion_matrix
from simplified_keras.transformations import predictions_to_classes, one_hot_to_sparse
import tensorflow as tf
import tensorflow.keras.callbacks as clb
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


def get_default_callbacks(model_path: Path,
                          monitor: str = 'val_acc',
                          base_patience: int = 3,
                          lr_reduce_factor: float = 0.5,
                          min_lr: float = 1e-7,
                          verbose: int = 1,
                          log_dir: Path = None,
                          gradients: bool = True, #change to false
                          confusion_matrix: bool = True, #change to false
                          loss: Callable = None,
                          data: Tuple[np.ndarray, np.ndarray] = None,
                          classes: list = None,
                          heatmap_options: dict = None,
                          csv_logdir: Path = None,
                          csv_append: bool = False,
                          save_latest: bool = False):
    callbacks = [
        clb.ReduceLROnPlateau(monitor=monitor, factor=lr_reduce_factor, min_lr=min_lr, patience=base_patience, verbose=verbose),
        clb.EarlyStopping(monitor=monitor, patience=(2 * base_patience + 1), verbose=verbose),
        clb.ModelCheckpoint(monitor=monitor, filepath=model_path, save_best_only=True, verbose=verbose)
    ]
    if log_dir:
        callbacks.append(ExtendedTensorBoard(log_dir, gradients, confusion_matrix, loss, data, classes, heatmap_options))

    if csv_logdir:
        if csv_append:
            callbacks.append(clb.CSVLogger(csv_logdir, append=True))
        else:
            callbacks.append(clb.CSVLogger(csv_logdir))

    if save_latest:
        latest_path = model_path.parent / f'{model_path.stem}_latest{model_path.suffix}'
        callbacks.append(clb.ModelCheckpoint(monitor=monitor, filepath=latest_path))

    return callbacks


def restore_callbacks(callbacks: List[clb.Callback], best_monitor_value: float):
    for callback in callbacks:
        if isinstance(callback, (clb.ModelCheckpoint, clb.ReduceLROnPlateau)):
            if 'loss' in callback.monitor and best_monitor_value <= 0:
                raise ValueError(f'{callback.monitor} should be > 0')

            if 'acc' in callback.monitor and not 0 < best_monitor_value < 1:
                raise ValueError(f'{callback.monitor} should be between 0 and 1')
            callback.best = best_monitor_value


# TODO: Support generators
class ExtendedTensorBoard(clb.TensorBoard):
    def __init__(self,
                 log_dir: Path,
                 gradients: bool = True,
                 confusion_matrix: bool = True,
                 loss: Callable = None,
                 data: Tuple[np.ndarray, np.ndarray] = None,
                 classes: list = None,
                 heatmap_options: dict = None,
                 **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.loss = loss
        self.data = data
        self.classes = classes
        if not classes and data:
            self.classes = [i for i in range(len(data[1][0]))]
        self.heatmap_options = heatmap_options
        self.gradients = gradients
        self.confusion_matrix = confusion_matrix

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        if self.gradients:
            assert self.loss and self.data, 'You have to pass loss and data arguments'
            self.__log_gradients(epoch)
        if self.confusion_matrix:
            assert self.data, 'You have to pass data argument'
            self.__log_confusion_matrix(epoch)
        self._train_writer.flush()

    def __log_gradients(self, epoch):
        x, y = self.data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_weights)
            z = self.model(x, training=True)
            loss = self.loss(y, z)
        grads = {w.name: tape.gradient(loss, w) for w in self.model.trainable_weights}

        with self._train_writer.as_default():
            for name, g in grads.items():
                mean = tf.reduce_mean(tf.abs(g))
                tf.summary.scalar(f"gradients_mean_{name}", mean, step=epoch)
                tf.summary.histogram(f"gradients_histogram_{name}", g, step=epoch)

    def __log_confusion_matrix(self, epoch):
        x, y = self.data
        test_pred = predictions_to_classes(self.model.predict(x))

        _, cm_normalized = get_confusion_matrixes(test_pred, one_hot_to_sparse(y))
        fig = plot_confusion_matrix(cm_normalized, self.classes, show=False)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        with self._train_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)


class WarmUpLR(clb.Callback):
    def __init__(self,
                 start_lr: float,
                 final_lr: float,
                 warmup_epochs: int,
                 verbose=0):

        super().__init__()
        if start_lr >= final_lr:
            raise ValueError('start_lr should be >= final_lr', f' Received: start_lr={start_lr}, final_lr{final_lr}')

        if warmup_epochs <= 0:
            raise ValueError('warmup_epochs should be > 0', f' Received: warmup_epochs={warmup_epochs}')

        self.current_step = 0
        self.start_lr = start_lr
        self.verbose = verbose
        self.final_lr = final_lr
        self.epoch = 0
        self.warmup_epochs = warmup_epochs
        self.should_update = True

    def on_batch_begin(self, batch, logs=None):
        if self.should_update:
            if self.current_step < self.warmup_steps:
                lr = self.warmup(current_step=self.current_step)
            else:
                lr = self.final_lr
                self.should_update = False

            K.set_value(self.model.optimizer.lr, lr)
            self.current_step += 1

            if self.verbose > 0:
                print(f'\nSetting learning rate to {lr}')

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.warmup_steps = int(self.warmup_epochs * self.params['samples'] / self.params['batch_size'])
        self.slope = self.final_lr / self.warmup_steps

    def warmup(self, current_step):
        shift = self.start_lr
        learning_rate = self.slope * current_step + shift
        return learning_rate