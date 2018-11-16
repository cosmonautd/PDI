import os
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

input_path = "./all/images_training_rev1"
image_list = os.listdir(input_path)
image_list.sort()

for image_name in image_list[:100]:

    image = cv2.imread(os.path.join(input_path, image_name),cv2.IMREAD_GRAYSCALE)
    detections = image.copy()

    t, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # _, otsu = cv2.threshold(image, 1*t, 255, cv2.THRESH_BINARY)

    d_transform = scipy.ndimage.distance_transform_edt(otsu)
    d_transform = cv2.GaussianBlur(d_transform, (9,9), 0)
    # d_transform = skimage.exposure.equalize_adapthist(d_transform)
    # d_transform = skimage.morphology.opening(d_transform, skimage.morphology.disk(11))

    _, otsu2 = cv2.threshold(image, 2.8*t, 255, cv2.THRESH_BINARY)
    local_max = skimage.feature.peak_local_max(otsu2, indices=False, min_distance=10, labels=otsu)
    markers = scipy.ndimage.label(local_max, structure=numpy.ones((3, 3)))[0]
    labels = skimage.morphology.watershed(-d_transform, markers, mask=otsu)
    segmentation = skimage.color.label2rgb(labels, bg_label=0)

    for label in numpy.unique(labels):
        if label > 0:
            object_ = labels.copy()
            object_[object_!=label] = 0
            object_props = skimage.measure.regionprops(object_)[0]
            if object_props.area > 10*10:
            # if object_props.area > 50*50:
                r0, c0 = object_props.bbox[0], object_props.bbox[1]
                h, w = object_props.bbox[2] - r0, object_props.bbox[3] - c0
                rr, cc = rectangle_perimeter(r0, c0, h, w)
                rr[rr>=424] = 424 - 1
                cc[cc>=424] = 424 - 1
                detections[rr, cc] = 255
    show_image([image, segmentation, detections])
    plt.show()