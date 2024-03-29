import pyspark
from pyspark.ml import Pipeline
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import split, explode
from pyspark.sql.functions import desc
from pyspark.sql.functions import regexp_replace, concat_ws
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql import functions
from pyspark.sql.types import *
import argparse
import glob
import json
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing tweets')
    parser.add_argument('--input_pagerank_id_file', type=str, default="/home/vt/extra_storage/Production/data/production_data_in/id_name_by_pagerank.csv",
                    help='input file that has the top 5000 pagerank ids')
    parser.add_argument('--input_tweets_folder', type=str, default="/home/vt/extra_storage/twitter_data/tweets_central_figures/*.txt",
                    help='The folder with the tweets')
    parser.add_argument('--output', type=str, default="/home/vt/extra_storage/Production/data/temp_data_in/tweets_top5000_joined_by_rank",
                    help='The output file for merged tweets')
    
    args = parser.parse_args()
    print(args)
    pagerank_id_file = args.input_pagerank_id_file
    input_tweets_folder = args.input_tweets_folder
    output_file = args.output

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
    
    pub_extracted = data.rdd.map(lambda x: ( x['user']['screen_name'], 
                                            x['user']['profile_image_url'],
                                            x['user']['statuses_count'],
                                            x['user']['created_at'],
                                            x['user']['description'],
                                            x['user']['id'], 
                                            x['id'], 
                                            x['full_text']) ).toDF(['name', 
                                                                    'profile_image_url',
                                                                    'statuses_count',
                                                                    'created_at',
                                                                    'description',
                                                                    'user_id',
                                                                    'tweet_id',
                                                                    'text'])
    pub_extracted_unique = pub_extracted.dropDuplicates(['tweet_id'])
    print("Count of extracted tweets ",pub_extracted.count())
    
    pagerank_top5000_ids = spark.read.csv(pagerank_id_file)
    
    pagerank_top5000_ids = pagerank_top5000_ids.withColumnRenamed('_c0','id').withColumnRenamed('_c1','username')
    pzip = pagerank_top5000_ids.rdd.zipWithIndex()
    pzip_DF = pzip.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(['userid','username','rank'])
    
    tweets_top5000_joined_by_rank = pzip_DF.join(pub_extracted, pzip_DF.userid == pub_extracted.user_id, 'full')
    
    print(tweets_top5000_joined_by_rank.show())
    
    spark.sql('DROP TABLE IF EXISTS extended_tweets_top5000_by_pagerank')
    
    tweets_top5000_joined_by_rank.write.option('path', "../output/extended").mode('overwrite').save('extended_tweets_top5000_joined_by_pagerank.json')
    
    tweets_top5000_joined_by_rank.write.option('path', "../data/spark-warehouse-files/extended_tweets_top5000_joined_by_pagerank").mode('overwrite').saveAsTable('extended_tweets_top5000_joined_by_pagerank')
    
    
    print(spark.catalog.listTables())



