from pyspark.ml.clustering import KMeans

# $example off$

from pyspark.sql import SparkSession

"""
An example demonstrating k-means clustering.
Run with:
  bin/spark-submit examples/src/main/python/ml/kmeans_example.py
This example requires NumPy (http://www.numpy.org/).
"""

spark=SparkSession.appName("KMeansExample").getOrCreate()

# $example on$
# Loads data.
dataset = spark.read.format("libsvm").load("user/uism172/ML/kmeans_data.txt")

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

centers = model.clusterCenters()
len(centers)

transformed = model.transform(dataset).select("features", "prediction")
rows = transformed.collect()



spark.stop()