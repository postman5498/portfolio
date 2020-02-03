#""SimpleApp.py"""
from pyspark import SparkContext
import os
import sys
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
print(cwd)


logFile = "sample_csv.csv"  # Should be some file on your system
sc = SparkContext("local", "Simple App")
logData = sc.textFile(logFile).cache()

print(logData)

numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()
numtexts = logData.filter(lambda s: 'text' in s).count()

print ("Lines with a: %i, lines with b: %i, lines with the word text: %i" % (numAs, numBs, numtexts))

time.sleep(30)