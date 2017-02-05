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

    print("Successfully imported Spark Modules")

except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)

# Initialize SparkContext
sc = SparkContext('local')

def mod(x):
    import numpy as np
    return (x, np.mod(x, 2))

rdd = sc.parallelize(range(10000)).map(mod).take(10)
print(rdd)