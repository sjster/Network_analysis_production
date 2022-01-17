import pyspark
from pyspark.ml import Pipeline
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.ml.feature import * 
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import split, explode
from pyspark.sql.functions import * 
from pyspark.sql.functions import regexp_replace, concat_ws
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql import functions
from pyspark.sql.types import *
import argparse
import glob
import json
import pickle

def write_vocab_csv(l):
  with open('/home/vt/extra_storage/Production/output/vocab.txt', 'w') as f:
        for elem in l:
            f.write(elem + "\n")

def custom_stop_words():
    stopwordList = ["rt"] 
    stopwordList.extend(StopWordsRemover().getStopWords())
    stopwordList = list(set(stopwordList))#optional
    return(stopwordList)
    
input_tweets_folder = "/home/vt/extra_storage/Production/output/tweets_top5000_joined_by_rank.json"

conf = pyspark.SparkConf()
conf.set('spark.local.dir', '/home/vt/extra_storage/')
conf.set('spark.sql.shuffle.partitions', '2100')
conf.set('spark.driver.maxResultSize', '8g')
conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
SparkContext.setSystemProperty('spark.executor.memory', '48g')
SparkContext.setSystemProperty('spark.driver.memory', '48g')
    
sc = SparkContext(appName='mm_exp', conf=conf)
spark = SparkSession(sc)

data = spark.read.option("header", "false").option("multiline",False).json(input_tweets_folder)
data = data.na.drop()
data = data.withColumn('text_cleaned', regexp_replace('text', '[#|,|&|!|~|*]|http.*', ''))

# ------------------- Grouping individual texts ---------------------- #

grouped_df = data.groupby('name').agg(collect_list('text_cleaned').alias("text_aggregated"))
grouped_appended_df = grouped_df.withColumn("text_aggregated_1", concat_ws(". ", "text_aggregated"))

# -------------------- Write out grouped data -------------------------------- #

grouped_appended_df.select("name", col("text_aggregated_1").alias("text")).repartition(1).write.save(path='/home/vt/extra_storage/Production/output/tweets_grouped.txt', format='json', mode='overwrite', sep=" ")

grouped_appended_df.filter(col('name') == 'LanaLokteff').select("name", col("text_aggregated_1").alias("text")).repartition(1).write.save(path='/home/vt/extra_storage/Production/output/tweets_LanaLokteff.txt', format='json', mode='overwrite', sep=" ")

grouped_appended_df.filter(col('name') == 'Lauren_Southern').select("name", col("text_aggregated_1").alias("text")).repartition(1).write.save(path='/home/vt/extra_storage/Production/output/tweets_Lauren_Southern.txt', format='json', mode='overwrite', sep=" ")

grouped_appended_df.filter(col('name') == 'Steve_Sailer').select("name", col("text_aggregated_1").alias("text")).repartition(1).write.save(path='/home/vt/extra_storage/Production/output/tweets_Steve_Sailer.txt', format='json', mode='overwrite', sep=" ")

grouped_appended_df.filter(col('name') == 'BrittPettibone').select("name", col("text_aggregated_1").alias("text")).repartition(1).write.save(path='/home/vt/extra_storage/Production/output/tweets_BrittPettibone.txt', format='json', mode='overwrite', sep=" ")

grouped_appended_df.filter(col('name') == 'RichardBSpencer').select("name", col("text_aggregated_1").alias("text")).repartition(1).write.save(path='/home/vt/extra_storage/Production/output/tweets_RichardBSpencer.txt', format='json', mode='overwrite', sep=" ")

# ------------------ Write out non-grouped data -------------------------------- #

data.filter(col('name') == 'LanaLokteff').select("name", col("text_cleaned").alias("text")).repartition(1).write.save(path='/home/vt/extra_storage/Production/output/tweets_nongrouped_LanaLokteff.txt', format='json', mode='overwrite', sep=" ")