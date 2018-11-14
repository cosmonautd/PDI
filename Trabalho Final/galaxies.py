import os
import cv2

image_name = "100008.jpg"
input_path = "./all/images_training_rev1"
image = cv2.imread(os.path.join(input_path, image_name), cv2.IMREAD_GRAYSCALE)

