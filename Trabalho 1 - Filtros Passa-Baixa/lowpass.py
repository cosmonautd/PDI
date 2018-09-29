import os
import cv2
import numpy
import matplotlib.pyplot as plt
import seaborn

import warnings
warnings.filterwarnings("ignore")

input_path = './images'
output_path = './output'

if not os.path.exists(output_path):
    os.mkdir(output_path)

def meanfilter_3x3(image):
    return cv2.blur(image, (3,3))

def meanfilter_5x5(image):
    return cv2.blur(image, (5,5))

def meanfilter_7x7(image):
    return cv2.blur(image, (7,7))

def meanfilter_custom_3x3(image):
    k = (1/16)*numpy.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    return cv2.filter2D(image, -1, k)

def medianfilter_3x3(image):
    return cv2.medianBlur(image, 3)

def medianfilter_5x5(image):
    return cv2.medianBlur(image, 5)

def medianfilter_7x7(image):
    return cv2.medianBlur(image, 5)

def histogram(path, image):
    x = numpy.array([p for row in image for p in row])
    seaborn.distplot(x, bins=64, kde=False, norm_hist=True)
    plt.xlim([-5, 260])
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

dataset = dict()

for image_name in os.listdir(input_path):
    id_ = image_name.split('.')[0]
    dataset[id_] = cv2.imread(os.path.join(input_path, image_name), cv2.IMREAD_GRAYSCALE)
    histogram(os.path.join(output_path, '_'.join([id_, 'original']))+'.pdf', dataset[id_])

functions = [meanfilter_3x3, meanfilter_5x5, meanfilter_7x7, meanfilter_custom_3x3,
                medianfilter_3x3, medianfilter_5x5, medianfilter_7x7]

for id_ in dataset:
    for f in functions:
        output = f(dataset[id_])
        cv2.imwrite(os.path.join(output_path, '_'.join([id_, f.__name__]))+'.jpg', output)
        histogram(os.path.join(output_path, '_'.join([id_, f.__name__]))+'.pdf', output)