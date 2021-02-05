print("++++++++++++++++++++++++ Import packages ++++++++++++++++++++++")
# System packages
import io
from io import StringIO
import os
import sys
# Table packages
import numpy as np
import pandas as pd
# Images preprocessing packages
from PIL import Image
from resizeimage import resizeimage
import cv2
# Pyspark packages
from pyspark import SparkContext
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import Row, SparkSession 
from sklearn import preprocessing
from pyspark.mllib.clustering import KMeans
# Write on S3
import boto.s3
import boto3

def get_path(test_train_set):
    """ Get the path of all images parent directories returned in a string. 
    
    The parameter test_train_set is a string with the values '/Training' or 
    '/Test/' to choose between the two sets. The global variable LOCAL indicate
    if the path has to be found on the local machine or on S3. The global 
    variable FOLDER indicated the main folder where two sets are. The returned 
    value is a string with the path to all the directories where the images are, 
    separated by a coma.
    """
    
    print("--------- Get the Paths ---------")
    image_path = ''
    if LOCAL:
        image_folder = FOLDER + test_train_set
        for root, directories, files in os.walk(image_folder):
            for file in directories:
                sub_folder = os.path.join(root, file).split('/')[-1]
                image_path = image_path + image_folder + sub_folder + ','
    else:
        image_folder = 's3://my-personal-bucket/' + FOLDER + test_train_set
        conn = boto.s3.connect_to_region('eu-west-1')
        bucket = conn.get_bucket("fruits-millet")
        folders = bucket.list(prefix=FOLDER + test_train_set, delimiter='/')
        for folder in folders:
            image_path = image_path + image_folder + folder.name.split("/")[-2] + ','
    image_path = image_path[0:-1]
    return image_path


def rescale_and_get_descriptors(image):
    """ Map the binaryFiles inputs, returns the category and descriptor.
    
    Transform the binary file into image. If the image is not a square, add a 
    white space to complete it. Thus rescale it to 100*100 pixels-square size. 
    Then apply orb openCV descriptors detection and returns the name of the 
    category as key and the list of descriptors as value.
    """ 
    
    try:
        # From bytes to image
        name, img = image
        image_img = Image.open(io.BytesIO(img))
        category = name.split('/')[-2]
        # Crop image
        fill_color=(255, 255, 255, 0)
        x, y = image_img.size
        if x < y:
            size_x = y
            size_y = y
        else:
            size_x = x
            size_y = x
        new_im = Image.new('RGB', (size_x, size_y), fill_color)
        new_im.paste(image_img, (int((size_x - x) / 2), int((size_y - y) / 2)))
        img_cropped = new_im
      # Rescale image
        width = 100
        img_rescaled = resizeimage.resize_cover(img_cropped, [width, width])
       # Convert to numpy
        np_img = np.array(image_img)
        # Get descriptors
        orb = cv2.ORB_create(nfeatures=50)
        keypoints_orb, desc = orb.detectAndCompute(np_img, None)
        # Check if none arefound
        if desc is None:
            cat = np.full(1, category)
        else:
            cat = np.full(desc.shape[0], category)
    except:
        cat = np.full(1, "error")
        desc = None
    return cat, desc


def get_categories_descriptors(IMAGE_PATH):
    """ Load all images, gives a list of categories and a RDD list of descriptors.
    
    The function loads the images through binaryFiles thanks to the string given
    by image_path. Maps the files with the rescale_and_get_descriptors function. 
    Thus removes the keys where no descriptors were found. Thus list the array 
    of descriptors. Then flatmap separately the descriptors and the categories
    to have lists of them. The returned value are the collected list of 
    categories and a RDD of list of descriptors not collected. 
    """
    
    print("--------- Find descriptors ---------")
    # Takes all images and names
    images = sc.binaryFiles(image_path, minPartitions = MIN_PARTITION)
    # Rescale all images and get descriptors. return (category, array of descriptors)
    tuples_decriptors = images.map(lambda img: rescale_and_get_descriptors(img))
    # Removes the descriptors which return nothing
    descriptors_filtered = tuples_decriptors.filter(lambda x: x[1] is not None)
    # List the descriptors for each category
    desc_cat = descriptors_filtered\
    .map(lambda x: (Row(fileName=x[0], features=x[1].tolist()))).cache()
    # Flat the descriptors and categories
    flat_desc = desc_cat.flatMap(lambda x: x['features'])
    flat_cat = desc_cat.flatMap(lambda x: x['fileName'])
    # Collect only the categories
    cat_collected = flat_cat.collect()
    return flat_desc, cat_collected


def run_kmeans(flat_desc, cat_collected):
    """  Returns kmeans model with 10 clusters per category and 1000 iterations."""
    
    print("--------- Begin kmeans ---------")
    # Select number of clusters
    K = len(np.unique(cat_collected)) * 10
    # Train the model
    model = KMeans\
    .train(flat_desc, K, maxIterations=1000, initializationMode="random")
    return model


def predict_kmeans(model, values):
    """ Predict the clusters with the model and values given."""
    
    # Get the results of clustering
    transformed = model.predict(values)
    # Collect
    predictions = transformed.collect()
    return predictions


def bag_word_creation(cat_collected, predictions):
    """ Creates a bag of words with the fruits names and the clusters predicted.
    
    Converts the label fruits names into numeric column name 'id'. Then 
    parallelize the data with a spark dataframe. This table is converted into 
    RDD to apply a map and ReduceByKey to get a string of clusters present for 
    each fruit. A second map convert this string into a list of words. Then the
    bag containing the number of cluster for each fruit is returned.
    """
    
    print("--------- Begin bag of words ---------")
    # Label the categories
    le = preprocessing.LabelEncoder()
    label_categories = le.fit_transform(cat_collected)
    # Concatenate the categories and prediction
    numpy_arr = np.concatenate((np.array(label_categories).reshape(-1,1), 
                                np.array(predictions).reshape(-1,1)), axis=1)
    pandas_df= pd.DataFrame(numpy_arr, columns=['id', 'prediction'])
    # Convert into spark DataFrame
    spark = SparkSession(sc)
    spark_df = spark.createDataFrame(pandas_df)
    # List all clusters for each category
    rdd_words = spark_df.select('id', 'prediction')\
    .rdd.map(lambda x:x)\
    .reduceByKey(lambda a,b: str(a) + ',' + str(b))
    # From that string of clusters/words we get a list of words
    df_words = rdd_words\
    .map(lambda tupl_words: (tupl_words[0], str(tupl_words[1]).split(',')))\
    .toDF(['category','words'])
    # Creates a vector from the count of words for each category
    vectorizer = CountVectorizer(inputCol="words", outputCol="bag_of_words")
    vectorizer_transformer = vectorizer.fit(df_words)
    bag_of_words = vectorizer_transformer\
    .transform(df_words)\
    .select('category', 'bag_of_words')
    return bag_of_words


def save_bag(bag_of_words, cat_collected, name):
    """ Save the bag of word locally or on S3"""
    
    print("--------- Save the bag of words ---------")
    # Transform into Pandas to save the bag of visual words
    pandaDf = bag_of_words.toPandas()
    pandaDf["category"] = np.unique(cat_collected)
    # Save the DataFrame
    if LOCAL:
        pandaDf.to_csv(name, index=False)
    else:
        csv_buffer = StringIO()
        pandaDf.to_csv(csv_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object('fruits-millet', name)\
        .put(Body=csv_buffer.getvalue(), ACL='public-read')


if __name__== "__main__": 
    
    try:
        if sys.argv[1]=='True': 
            LOCAL=True
        elif sys.argv[1]=='False':
            LOCAL=False
        else: 
            raise Exception
        FOLDER = sys.argv[2]
        MIN_PARTITION = int(sys.argv[3])
        NAME_OUT_TRAIN = sys.argv[4]
        NAME_OUT_TEST = sys.argv[5]
    except:
        print("Usage: spark-submit fruits_reduce.py <local mode(True or False)>\
        <image main folder('fruits' or 'fruits2')> <partitions> <name csv \
        out train set> <name csv out test set>")
        sys.exit(0)
        

    sc = SparkContext()

    print("++++++++++++++++++++++++ BEGIN TRAINING SET ++++++++++++++++++++++")
    image_path = get_path('/Training/')
    flat_desc, cat_collected = get_categories_descriptors(image_path)
    model = run_kmeans(flat_desc, cat_collected)
    predictions = predict_kmeans(model, flat_desc)
    bag_of_words = bag_word_creation(cat_collected, predictions)
    save_bag(bag_of_words, cat_collected, NAME_OUT_TRAIN)

    print("++++++++++++++++++++++++ BEGIN TESTING SET ++++++++++++++++++++++")
    image_path = get_path('/Test/')
    flat_desc, cat_collected = get_categories_descriptors(image_path)
    predictions = predict_kmeans(model, flat_desc)
    bag_of_words = bag_word_creation(cat_collected, predictions)
    save_bag(bag_of_words, cat_collected, NAME_OUT_TEST)

    print("DONE")