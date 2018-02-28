import sys
from operator import add
from pyspark import SparkConf, SparkContext

sc=SparkContext()
filename="./ANA/"+sys.argv[1]
rdd=sc.textFile(filename)
rdd2=rdd.flatMap(lambda x:x.split(' ')).map(lambda x:(x,1)).reduceByKey(add)
output=rdd2.collect()

for (word,count) in output:
 print("%s: %i" % (word, count))

