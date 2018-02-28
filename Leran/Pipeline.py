from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf,mean
from pyspark.sql.types import DoubleType,StringType,IntegerType,FloatType
import re
from pyspark.ml.feature import StringIndexer,Bucketizer,VectorAssembler,Normalizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics


#pyspark --packages com.databricks:spark-csv_2.11:1.3.0


//Categorize a passenger as child if his/her age is less than 15 
//(more chances of survival)



addChild = udf((lambda sex,age: 'child' if age<15 else sex),StringType())

//withFamily is true(1) if the family size excluding self is > 3 
//(large family may have more/less chance of survival)

withFamily = udf((lambda sib,par: 1.0 if (sib + par > 3) else 0.0),DoubleType())



sc= SparkContext()
sqlContext=SQLContext(sc)
path="/user/uism172/ML/train.csv"
train=sqlContext.load(source="com.databricks.spark.csv", path = 'ML/train.csv',header = True,inferSchema = True)

#Ignore/drop the rows having missing values. 
train.na.drop()
avgAge=train.select(mean("Age").cast("float").alias("avgAge")).first()
avgAge.avgAge
train_data = train.na.fill(avgAge.avgAge,("Age"))
train_embarked_filled = train_data.na.fill("S",("Embarked"))

temp  = train_data.withColumn("test",train_data["PassengerId"])
temp = train_data.withColumn("test",train_data["PassengerId"]-1)
temp.select("PassengerId","test").show(3)

p = re.compile('(Dr|Miss|Mrs?|Ms|Miss|Master|Rev|Capt|Mlle|Col|Major|Sir|Lady|Mme|Don)')  
def findTitles(Name):
   title=p.search(Name).group()
   if ((title == "Don") or (title =="Major") or (title =="Capt")):
    return "Sir."
   if ((title =="Mlle") or (title =="Mme")):
    return "Miss."
   return title
   
  
func = udf(findTitles,StringType())
train_data =train_data.withColumn("Title",func("Name"))
train_data = train_data.withColumn("Sex", addChild('Sex','Age'))

//for converting integer columns to double. Requires since few of the 
//columns of our DataFrame are of Int type.
train_data=train_data.withColumn("Pclass",train_data["Pclass"].cast(DoubleType()))
train_data=train_data.withColumn("Survived",train_data["Survived"].cast(DoubleType()))
train_data = train_data.withColumn("Family", withFamily("SibSp","Parch"))    


//execution of fit() and transform() will be done by the pipeline, this is shown to explain how fit and transform works

titleInd = StringIndexer(inputCol="Title",outputCol="TitleIndex")
strIndModel = titleInd.fit(train_data)
strIndModel.transform(train_data)
sexInd =  StringIndexer(inputCol="Sex",outputCol="SexIndex")
fareSplits = [0.0,10.0,20.0,30.0,40.0,float("inf")]
fareBucketize = Bucketizer(splits=fareSplits, inputCol="Fare", outputCol="FareBucketed")


assembler = VectorAssembler(inputCols=["SexIndex","Age","TitleIndex", "Pclass", "Family","FareBucketed"],outputCol='features_temp')
normalizer = Normalizer(inputCol="features_temp",outputCol="features")

lr = LogisticRegression(maxIter=10)
lr.setLabelCol("Survived")
pipeline = Pipeline().setStages([sexInd, titleInd, fareBucketize, assembler, normalizer,lr])

splits = train_data.randomSplit([0.8, 0.2], seed = 11L)
traind = splits[0].cache()
test = splits[1].cache()

model = pipeline.fit(traind)
result = model.transform(test)
result = result.select("prediction","Survived")
predictionAndLabels = result.rdd
metrics = BinaryClassificationMetrics(predictionAndLabels)
print(metrics.areaUnderROC)

//train the model
model = pipeline.fit(train_data)


//Test data
submission_data=sqlContext.load(source="com.databricks.spark.csv", path = 'ML/test.csv',header = True,inferSchema = True)
avgAge=train.select(mean("Age").cast("double").alias("avgAge")).first()
submission_data = submission_data.na.fill(avgAge, "Age")

submission_data = submission_data.withColumn("Title",func("Name"))
submission_data = submission_data.withColumn("Sex", addChild('Sex','Age'))

//for converting integer columns to double. Requires since few of the 
//columns of our DataFrame are of Int type.
submission_data = submission_data.withColumn("Pclass",submission_data["Pclass"].cast(DoubleType()))
submission_data = submission_data.withColumn("Family", withFamily("SibSp","Parch"))
submission_data = submission_data.withColumn("Survived",submission_data["PassengerId"].cast(DoubleType()))
result = model.transform(submission_data)
result.select("PassengerId","prediction").show(3)

