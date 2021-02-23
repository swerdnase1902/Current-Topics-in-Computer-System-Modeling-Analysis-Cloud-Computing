from pyspark.sql import SparkSession
import numpy as np
from PIL import Image
import cv2
import io
import imutils

def stitch(img_pair):
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    ret = (status, stitched) = stitcher.stitch([img_pair[0], img_pair[1]])
    return ret

def binary_to_cv(binaryfile):
    img = np.array(Image.open(io.BytesIO(binaryfile[1])).convert('RGB'))[:, :, ::-1].copy()
    return img


if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("PythonSimpleStitch") \
        .getOrCreate()
    sc = spark.sparkContext
    left_images = sc.binaryFiles('example_img/left', minPartitions=sc.defaultMinPartitions)
    left_images = left_images.map(lambda img: binary_to_cv(img))
    right_images = sc.binaryFiles('example_img/right', minPartitions=sc.defaultMinPartitions)
    right_images = right_images.map(lambda img: binary_to_cv(img))

    pairs = left_images.zip(right_images)
    processed = pairs.map(lambda p: stitch(p))
    processed = processed.collect()
    for (status, img) in processed:
        if status==0:
            cv2.imshow('stitched image', img)
            cv2.waitKey(0)
    exit(0)
