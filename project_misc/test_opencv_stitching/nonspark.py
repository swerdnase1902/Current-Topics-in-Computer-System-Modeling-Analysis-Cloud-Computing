from pyspark.sql import SparkSession
import numpy as np
from PIL import Image
import cv2
import io
import imutils

def stitch(img_pair):
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    imgs = [img_pair[0], img_pair[1]]
    (status, stitched) = stitcher.stitch(imgs)
    if status==0:
        return stitched
    else:
        return None

def binary_to_cv(binaryfile):
    img = np.array(Image.open(open(binaryfile, 'rb')).convert('RGB'))[:, :, ::-1].copy()
    return img

if __name__ == '__main__':
    img1 = open('example_img/left/000.jpg', 'rb')
    cv_img1 = binary_to_cv('example_img/left/000.jpg')
    img2 = open('example_img/right/000.jpg', 'rb')
    cv_img2 = binary_to_cv('example_img/right/000.jpg')
    processed = stitch((cv_img1, cv_img2))
    exit(0)