from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.ml.feature import Bucketizer

sc = SparkContext()
sqlContext = SQLContext(sc)

splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]
data = [(-999.9,), (-0.5,), (-0.3,), (0.0,), (0.2,), (999.9,)]

dataFrame = sqlContext.createDataFrame(data, ["features"])
bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bucketedFeatures")
bucketedData = bucketizer.transform(dataFrame)
print("Bucketizer output with %d buckets" % (len(bucketizer.getSplits()-1)))

bucketedData.show()
+--------+----------------+
|features|bucketedFeatures|
+--------+----------------+
|  -999.9|             0.0|
|    -0.5|             1.0|
|    -0.3|             1.0|
|     0.0|             2.0|
|     0.2|             2.0|
|   999.9|             3.0|
+--------+----------------+
