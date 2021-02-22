from pyspark.sql import SparkSession
import numpy as np
from PIL import Image
import cv2 as cv
import io


def stitch(img_pair):
    # https://stackoverflow.com/questions/60194755/opencv-read-images-from-pyspark-and-pass-to-a-keras-model
    # name, img = binary_images
    # pil_image = Image.open(io.BytesIO(img)).convert('RGB')
    # cv2_image = numpy.array(pil_image)
    # cv2_image = cv2_image[:, :, ::-1].copy()
    _, left_img = img_pair[0]
    left_img = np.array(Image.open(io.BytesIO(left_img)).convert('RGB'))[:, :, ::-1].copy()
    _, right_img = img_pair[1]
    right_img = np.array(Image.open(io.BytesIO(right_img)).convert('RGB'))[:, :, ::-1].copy()

    return 'Hello'

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("PythonSimpleStitch") \
        .getOrCreate()
    sc = spark.sparkContext
    left_images = sc.binaryFiles('example_img/left', minPartitions=sc.defaultMinPartitions)
    right_images = sc.binaryFiles('example_img/right', minPartitions=sc.defaultMinPartitions)

    pairs = left_images.zip(right_images)
    processed = pairs.map(lambda p: stitch(p))
    exit(0)
