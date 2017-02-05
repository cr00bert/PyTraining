import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME'] = "C:\\Spark\\spark-2.1.0-bin-hadoop2.7"

# Append pyspark  to Python Path
sys.path.append("C:\\Spark\\spark-2.1.0-bin-hadoop2.7\\python")
sys.path.append("C:\\Spark\\spark-2.1.0-bin-hadoop2.7\\python\\pyspark")
sys.path.append("C:\\Spark\\spark-2.1.0-bin-hadoop2.7\\python\\lib\\py4j-0.10.4-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql.types import StringType
    from pyspark import SQLContext

    print("Successfully imported Spark Modules")

except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)

# Initialize SparkContext
conf = SparkConf().setAppName('BinomialLogRegression').setMaster('local')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

from pyspark.ml.classification import LogisticRegression

# Load training data
# training = sc.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
training = sqlContext.read.format("libsvm").load("sample_libsvm_data.txt")


lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fit the model
mlrModel = mlr.fit(training)

# Print the coefficients and intercepts for logistic regression with multinomial family
print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(mlrModel.interceptVector))
