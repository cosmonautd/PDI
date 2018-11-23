import os
import csv
import cv2
import numpy
import scipy.ndimage
import skimage.color
import skimage.filters
import skimage.feature
import skimage.measure
import skimage.exposure
import skimage.morphology
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import elmshape

import numpy
import pickle
import tflearn
import tensorflow
from sklearn.model_selection import StratifiedKFold

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

tflearn.init_graph(num_cores=4)

# Building 'VGG Network'
network = input_data(shape=[None, 28, 28, 1])

network = conv_2d(network, 14, 3, activation='relu')
network = conv_2d(network, 14, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=0.0001)

model = tflearn.DNN(network)
model.load('models/galaxies-cnn-04.model')

def show_image(images):
    """ Displays images on screen
    """
    n = len(images)
    if n == 1:
        fig, (ax0) = plt.subplots(ncols=1)
        ax0.imshow(images[0], cmap='gray', interpolation='bicubic')
        ax0.axes.get_xaxis().set_ticks([])
        ax0.axes.get_yaxis().set_visible(False)
    else:
        fig, axes = plt.subplots(ncols=n, figsize=(4*n, 4))
        for ax, image in zip(axes, images):
            ax.imshow(image, cmap='gray', interpolation='bicubic')
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_visible(False)
    fig.tight_layout()

def show_image_surface(image):
    """ Displays image as a surface
    """
    image = cv2.resize(image, (0,0), fx=0.1, fy=0.1) 
    xx, yy = numpy.mgrid[0:image.shape[0], 0:image.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx,yy,image,rstride=1,cstride=1,cmap=plt.cm.gray,linewidth=0)

def rectangle_perimeter(r0, c0, width, height, shape=None, clip=False):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + height, c0 + height]
    return skimage.draw.polygon_perimeter(rr, cc, shape=shape, clip=clip)

def square(r, c, h, w):
    h_ = h if w < h else w
    w_ = w if h < w else h
    r_ = r if w < h else max(r - (h_ - h)/2, 0)
    c_ = c if h < w else max(c - (w_ - w)/2, 0)
    return int(r_), int(c_), int(h_), int(w_)

solutions_path = "./all/training_solutions_rev1.csv"

dataset = dict()
with open(solutions_path) as csvfile:
    csvfile.readline()
    solutions = csv.reader(csvfile, delimiter=',')
    for s in solutions:
        id_ = s[0]
        p_ = [float(p) for p in s[1:3]]
        if max(p_) > 0.7:
            index_max = max(range(len(p_)), key=p_.__getitem__)
            class_ = index_max
            if not class_ in dataset.keys():
                dataset[class_] = dict()
                dataset[class_]['contours'] = list()
                dataset[class_]['features'] = list()
                dataset[class_]['galaxies'] = list()
            dataset[class_]['galaxies'].append((id_, class_))

input_path = "./all/images_training_rev1"
image_list = [dataset[class_]['galaxies'] for class_ in [0, 1]]
image_list = [item for sublist in image_list for item in sublist]

method = {'approach':['neighborhood','contour_portion'], 'params':[(2,4,6,8),(5,10,15,20)]}
stack = [elmshape.ContourDescriptor(mode=m[0], neurons=4, params=m[1]) \
            for m in zip(method['approach'], method['params'])]
descriptor = elmshape.StackedContourDescriptor(stack)

for (id_, class_) in random.sample(image_list, 100):

    # Image loading
    image = cv2.imread(os.path.join(input_path, id_ + ".jpg"))
    detections = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center = numpy.array(image.shape)/2

    # Image preprocessing
    t, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # _, otsu = cv2.threshold(image, 1*t, 255, cv2.THRESH_BINARY)

    # Objects segmentation
    d_transform = scipy.ndimage.distance_transform_edt(otsu)
    d_transform = cv2.GaussianBlur(d_transform, (9,9), 0)
    # d_transform = skimage.exposure.equalize_adapthist(d_transform)
    # d_transform = skimage.morphology.opening(d_transform, skimage.morphology.disk(11))
    _, otsu2 = cv2.threshold(image, 2.8*t, 255, cv2.THRESH_BINARY)
    local_max = skimage.feature.peak_local_max(otsu2, indices=False, min_distance=10, labels=otsu)
    markers = scipy.ndimage.label(local_max, structure=numpy.ones((3, 3)))[0]
    labels = skimage.morphology.watershed(-d_transform, markers, mask=otsu)
    segmentation = skimage.color.label2rgb(labels, bg_label=0)

    # Extract info from objects
    objects = list()
    for label in numpy.unique(labels):
        if label > 0:
            object_ = dict()
            object_["mask"] = labels.copy().astype(numpy.uint8)
            object_["mask"][object_["mask"]!=label] = 0
            object_["mask"][object_["mask"]==label] = 255
            object_["props"] = skimage.measure.regionprops(object_["mask"])[0]
            object_["centroid"] = object_["props"].centroid
            object_["center_distance"] = numpy.linalg.norm(object_["centroid"] - center)
            if object_["props"].area > 0*0:
                objects.append(object_)
    
    if len(objects) < 1: continue
    # Get objects of interest
    main_objects = [min(objects, key=lambda x:x['center_distance'])]
    # main_objects = objects

    # Mark objects of interest
    for object_ in main_objects:
        r, c = object_["props"].bbox[0], object_["props"].bbox[1]
        h, w = object_["props"].bbox[2] - r, object_["props"].bbox[3] - c
        r, c, w, h = square(r, c, h, w)
        rr, cc = rectangle_perimeter(r, c, h, w)
        rr[rr>=424] = 424 - 1
        cc[cc>=424] = 424 - 1
        detections[rr, cc] = numpy.array([255, 255, 255])
    
    # Get features from objects of interest
    for object_ in main_objects:
        r, c = object_["props"].bbox[0], object_["props"].bbox[1]
        h, w = object_["props"].bbox[2] - r, object_["props"].bbox[3] - c
        r, c, w, h = square(r, c, h, w)
        object_crop = image[r:r+h, c:c+w]
        conv_features = cv2.resize(object_crop, (28, 28))
        features = numpy.array([conv_features])
        features = features.reshape([-1, 28, 28, 1])

        y = model.predict(features)[0]
        class_pred = numpy.argmax(y)
        print(y)
    
        # Show image, segmentation and detected main objects
        font= cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText1 = (10, 380)
        bottomLeftCornerOfText2 = (10, 400)
        fontScale = 0.5
        fontColor = (255,255,255)
        lineType = 2

        cv2.putText(detections, "Real class: %d" % (class_), bottomLeftCornerOfText1, font, fontScale, fontColor, lineType)
        cv2.putText(detections, "Predicted class: %d" % (class_pred), bottomLeftCornerOfText2, font, fontScale, fontColor, lineType)

    show_image([image, segmentation, detections])
    plt.show()