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
#  with open('/home/vt/extra_storage/Production/output/vocab.txt', 'w') as f:
  with open('vocab.txt', 'w') as f:
        for elem in l:
            f.write(elem + "\n")

def custom_stop_words():
    stopwordList = ["rt"] 
    stopwordList.extend(StopWordsRemover().getStopWords())
    stopwordList = list(set(stopwordList))#optional
    return(stopwordList)
    
input_tweets_folder = "/home/vt/extra_storage/Production/output/tweets_top5000_joined_by_rank.json"
input_tweets_folder = "/home/vt/extra_storage/Production/output/tweets_nongrouped_LanaLokteff.txt"

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
#grouped_df = data.groupby('name').agg(collect_list('text_cleaned').alias("text_aggregated"))
#grouped_appended_df = grouped_df.withColumn("text_aggregated_1", concat_ws(". ", "text_aggregated"))
# --------------------------------------------------------------------#
tokenizer = Tokenizer(inputCol="text_cleaned", outputCol="words")
wordsData = tokenizer.transform(data)
stopwordlist = custom_stop_words()
remover = StopWordsRemover(inputCol="words", outputCol="filtered_col", stopWords=stopwordlist)
filtered = remover.transform(wordsData)
filtered.select("filtered_col").show(truncate=False)
cv = CountVectorizer(inputCol="filtered_col", outputCol="rawFeatures", vocabSize=15000, minDF=2.0)
cvmodel = cv.fit(filtered)

# ------------------------- Get vocabulary information  ----------------#

vocab = cvmodel.vocabulary # list of words in order, write out for vocab.txt file
d_bow = filtered.count()   # number of documents in the vocabulary
w_bow = len(vocab)         # number of words in the vocabulary

# ------------------------ Generate the BOW data structure ------------------------#

vector_udf = udf(lambda vector: vector.numNonzeros(), LongType())
vocab_broadcast = sc.broadcast(vocab)

# Example schema for rawFeatures
#              Filtered text                                        rawFeatures
# [@santi_abascal:, hipercor, miserable., olvidamos.] | (38,[5,14,26,36],[1.0,1.0,1.0,1.0])
# The first value is the length of the vocabulary, the second is an array of word indices, the third is the count # of the words in that second array
# rawFeatures has [index of word],[count of that word] arrays. index of word is from vocab array
featurized = cvmodel.transform(filtered) 
#nnz_bow = featurized.select(vector_udf('rawFeatures')).groupBy().sum().collect()

# Get the count of the words from the rawFeatures column
sparse_values = udf(lambda v: v.values.tolist(), ArrayType(DoubleType()))
# Get the indices of the words corresponding to the counts
sparse_indices = udf(lambda v: v.indices.tolist(), ArrayType(LongType()))

nnz = featurized.withColumn('vals', sparse_values('rawFeatures')).withColumn('indices', sparse_indices('rawFeatures')) 

# Creat zipped column from vals and indices
t = udf(lambda vals1, vals2: [(int(elem[0]), int(elem[1]) + 1) for elem in zip(vals1,vals2)], ArrayType(ArrayType(LongType())))
nnz = nnz.withColumn('zipped_array', t(col('vals'), col('indices')))

# Add an index column
nnz_indexed = nnz.select('zipped_array').rdd.zipWithIndex().toDF()
nnz_extracted = nnz_indexed.select(explode(col('_1').getItem('zipped_array')), col('_2'))

# Extract the column values and add the index value, write out the docword.txt file
nnz_extracted.select(col('_2') + 1, nnz_extracted['col'].getItem(1), nnz_extracted['col'].getItem(0)).repartition(1).write.save(path='docword.txt', format='csv', mode='overwrite', sep=" ")

#fzipped = nnz_elements.select('vals','indices').rdd.zipWithIndex().toDF() # vals is count of a word, indices is index of a word
#fzipped_sep = fzipped.withColumn('vals', fzipped['_1'].getItem("vals"))
#fzipped_sep = fzipped_sep.withColumn('indices', fzipped['_1'].getItem("indices"))
#fzipped_sep2 = fzipped_sep.select("_2", arrays_zip("indices", "vals"))
#nnz_data = fzipped_sep2.select("_2", explode("arrays_zip(indices, vals)"))
#out2 = nnz_data.withColumn('indices', out['col'].getItem("indices")).withColumn('cnt', out['col'].getItem("vals")).withColumn('reindexed', out2['_2'] + 1).select('reindexed', 'indices', col('cnt').cast(IntegerType()))
#nnz_bow = out2.select(sum(col('cnt'))).collect()[0][0] # number of nnz in the documents
# ------------------------- Write docword.txt file ------------------------#
#out2.repartition(1).write.save(path='docword.txt', format='csv', mode='overwrite', sep=" ")
# ------------------------- Write the vocab file --------------------------#
write_vocab_csv(vocab)


