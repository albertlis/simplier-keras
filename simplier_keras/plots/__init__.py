import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_with_img(i, predictions, true_label, img):
    predictions, true_label, img = predictions[i], true_label[i], img[i]
    predicted_label = np.argmax(predictions)
    true_value = np.argmax(true_label)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    plt.yticks(np.arange(len(true_label)))
    thisplot = plt.barh(range(len(predictions)), predictions, color="gray")
    thisplot[predicted_label].set_color('r')
    thisplot[true_value].set_color('g')

    plt.subplot(1, 2, 2)

    plt.imshow(img, cmap='gray')
    # plt.xlabel("{} {:2.0f}% ({})".format(true_label[predicted_label], 100 * np.max(predictions), true_label[true_value]))
    plt.show()


# grayscale only
def plot_gray_img_with_histogram(img, figsize=(10, 5), brightness_range=(0, 255)):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(cdf_normalized, color='b')
    ax1.hist(img.flatten(), 256, [0, 256], color='r')
    ax1.set_xlim([0, 256])

    vmin, vmax = brightness_range
    ax2.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)