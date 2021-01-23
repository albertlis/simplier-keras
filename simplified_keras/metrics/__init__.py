import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_confusion_matrixes(predicted_classes, labels):
    cm= tf.math.confusion_matrix(labels=labels, predictions=predicted_classes).numpy()
    cm_normalized= np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    return cm, cm_normalized


def get_model_statistics(confusion_matrix):
    class Stats:
        def __init__(self):
            self.FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
            self.FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            self.TP = np.diag(confusion_matrix)
            self.TN = confusion_matrix.sum() - (self.FP + self.FN + self.TP)
            # Sensitivity/true positive rate
            self.TPR = self.TP / (self.TP + self.FN)
            # Specificity/true negative rate
            self.TNR = self.TN / (self.TN + self.FP)
            # Precision/positive predictive value
            self.PPV = self.TP / (self.TP + self.FP)
            # Negative predictive value
            self.NPV = self.TN / (self.TN + self.FN)
            # Fall out or false positive rate
            self.FPR = self.FP / (self.FP + self.TN)
            # False negative rate
            self.FNR = self.FN / (self.TP + self.FN)
            # False discovery rate
            self.FDR = self.FP / (self.TP + self.FP)
            # Overall accuracy for each class
            self.ACC = (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)
            self.subplot_options = {'nrows': 6, 'ncols': 2, 'figsize': (20, 10)}
            self.heatmap_options = {'annot': True, 'cmap': plt.cm.Blues, 'vmin': 0}

        def visualize(self, labels):
            attributes = self.__dict__.keys()
            fig, axes  = plt.subplots(**self.subplot_options)
            for i, att in enumerate(attributes):
                attribute = getattr(self, att)
                attribute = attribute.reshape((1, len(labels)))
                df = pd.DataFrame(attribute, index=[att], columns=labels)
                xtick = False
                if i > len(attributes) - 3:
                    xtick = True
                if issubclass(attribute.dtype.type, np.integer):
                    sns.heatmap(df, xticklabels=xtick, ax=axes.flat[i], fmt='d', **self.heatmap_options)
                else:
                    sns.heatmap(df, xticklabels=xtick, ax=axes.flat[i], **self.heatmap_options)
            plt.show()
            return fig
    return Stats()


def get_folders_statistics(directory):
    class Stats:
        def __init__(self, info):
            self.info = dict(sorted(info.items(), key=lambda item: item[0]))
            self.nr_of_elements = sum(info.values())
            self.subplot_options = {'figsize': (10, 5)}
            self.xticks_options = {'rotation': 60, 'horizontalalignment': 'right', 'size': 8}
            self.yticks_options = {'size': 8}
            self.rcParams_options = {"axes.labelsize": 10}

        def bar_plot(self, labels_size=10):
            df = pd.DataFrame(self.info, index=[0])
            fig, ax = plt.subplots(**self.subplot_options)
            f = sns.barplot(data=df, ax=ax)
            f.set(xlabel='Classes', ylabel='Images')
            f.set_xticklabels(f.get_xticklabels(), **self.xticks_options)
            yticks = [int(tick) for tick in f.get_yticks()]
            f.set_yticklabels(yticks, **self.yticks_options)
            plt.rcParams.update(self.rcParams_options)
            return fig

    subfolders, counted_pictures = [], []

    for element in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, element)):
            subfolders.append(element)

    for folder in subfolders:
        path = os.path.join(directory, folder)
        nr_of_pictures = len(os.listdir(path))
        counted_pictures.append(nr_of_pictures)

    inf = {key: value for key, value in zip(subfolders, counted_pictures)}
    return Stats(inf)
