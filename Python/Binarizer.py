from pyspark.ml.feature import Binarizer
from pyspark.sql import SQLContext
from pyspark import SparkContext

sc = SparkContext()
sqlContext = SQLContext(sc)
df=sqlContext.createDataFrame([(0,0.1),(1,0.8),(2,0.2)],["id", "feature"])
binarizer = Binarizer(threshold=0.5, inputCol="feature", outputCol="binarized_feature")
binarizedDataFrame = binarizer.transform(df)
print("Binarizer output with Threshold = %f" % binarizer.getThreshold())
#Binarizer output with Threshold = 0.500000
binarizedDataFrame.show()
+---+-------+-----------------+
| id|feature|binarized_feature|
+---+-------+-----------------+
|  0|    0.1|              0.0|
|  1|    0.8|              1.0|
|  2|    0.2|              0.0|
+---+-------+-----------------+
