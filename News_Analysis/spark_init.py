from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row

conf = SparkConf().setAppName("word_cloud").setMaster("local[1]")
conf.set('spark.logConf', 'true')
conf.set('spark.executor.memory', '2g')
conf.set('spark.executor.cores', '2')
conf.set('spark.cores.max', '2')
conf.set('spark.driver.memory', '2g')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SQLContext(sc)

