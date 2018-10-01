import os
import cv2
import numpy
import matplotlib.pyplot as plt
import seaborn
import skimage.morphology

import warnings
warnings.filterwarnings("ignore")

input_paths = ["./imgs_pb", "./imgs_cinza"]
output_path = "./output"

if not os.path.exists(output_path):
    os.mkdir(output_path)

def erosion(image, kernel):
    return cv2.erode(image, kernel, iterations=1)

def dilation(image, kernel):
    return cv2.dilate(image, kernel, iterations=1)

def opening(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def square(size):
    return skimage.morphology.square(size)

def line(size):
    return skimage.morphology.rectangle(size,1)

def disk(size):
    return skimage.morphology.disk(numpy.floor(size/2))

dataset = dict()

for input_path in input_paths:
    for image_name in os.listdir(input_path):
        id_ = image_name.split(".")[0]
        dataset[id_] = cv2.imread(os.path.join(input_path, image_name), cv2.IMREAD_GRAYSCALE)

operations = [erosion, dilation, opening, closing]
structuring_elements = [square, line, disk]
sizes = [5, 7, 9]

for id_ in dataset:
    for op in operations:
        for st in structuring_elements:
            for sz in sizes:
                kernel = st(size=sz)
                output = op(dataset[id_], kernel)
                h, w = output.shape
                h, w = (int(0.99*h), int(0.99*w))
                output = skimage.transform.resize(output, (h, w), preserve_range=True)
                description = "_".join([id_, op.__name__, st.__name__, str(sz)])
                cv2.imwrite(os.path.join(output_path, description)+".jpg", output)