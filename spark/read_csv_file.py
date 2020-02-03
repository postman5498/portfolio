#""SimpleApp.py"""
from pyspark import SparkContext
import os
import sys
import time
from pyspark.sql import SQLContext
sc = SparkContext()
sqlContext = SQLContext(sc)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
print(cwd)


df = sqlContext \
    .read.format("com.databricks.spark.csv") \
    .option("header", "true")\
    .option("inferschema", "true")\
    .option("mode", "DROPMALFORMED")\
    .load("sample_csv.csv")

df.show(n=10)

#numAs = logData.filter(lambda s: 'a' in s).count()
#numBs = logData.filter(lambda s: 'b' in s).count()
#numtexts = logData.filter(lambda s: 'text' in s).count()

#print ("Lines with a: %i, lines with b: %i, lines with the word text: %i" % (numAs, numBs, numtexts))

for i in range (0,20):
    print(i)
    time.sleep(1)