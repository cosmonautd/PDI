import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import skimage.morphology
import skimage.segmentation
import skimage.color

from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from skimage.segmentation import slic

import warnings
warnings.filterwarnings("ignore")

input_paths = ["./imagens-cor-segmentacao"]
output_path = "./output"

if not os.path.exists(output_path):
    os.mkdir(output_path)

def watershed(image):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    distance = ndi.distance_transform_edt(thresh)
    local_maxi = peak_local_max(distance, labels=thresh, min_distance=10, indices=False)
    markers = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
    labels = skimage.segmentation.watershed(-distance, markers, mask=image)
    return skimage.color.label2rgb(labels, image, kind='avg')

def kmeans(image):
    labels = skimage.segmentation.slic(image, n_segments=500, compactness=1)
    return skimage.color.label2rgb(labels, image, kind='avg')

dataset = dict()

for input_path in input_paths:
    for image_name in os.listdir(input_path):
        id_ = image_name.split(".")[0]
        dataset[id_] = cv2.imread(os.path.join(input_path, image_name))
        dataset[id_] = cv2.pyrMeanShiftFiltering(dataset[id_], 21, 54)
        dataset[id_] = cv2.cvtColor(dataset[id_], cv2.COLOR_BGR2GRAY)

operations = [watershed]

for id_ in dataset:
    for op in operations:
        output = op(dataset[id_])
        description = "_".join([id_, op.__name__])
        cv2.imwrite(os.path.join(output_path, description)+".jpg", output)