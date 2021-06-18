import pyspark
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer, StopWordsRemover
from pyspark import SparkContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import concat_ws
from pyspark.sql.functions import collect_list


def custom_stop_words():
    stopwordList = ["rt"] 
    stopwordList.extend(StopWordsRemover().getStopWords())
    stopwordList = list(set(stopwordList))#optional
    return(stopwordList)

conf = pyspark.SparkConf()
conf.set('spark.local.dir', '/home/vt/extra_storage/')
conf.set('spark.sql.shuffle.partitions', '2100')
conf.set('spark.driver.maxResultSize', '8g')
conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
SparkContext.setSystemProperty('spark.executor.memory', '48g')
SparkContext.setSystemProperty('spark.driver.memory', '48g')
    
sc = SparkContext(appName='mm_exp', conf=conf)
spark = SparkSession(sc)
    
input_tweets_folder = "/home/vt/extra_storage/Production/output/tweets_top5000_joined_by_rank.json"
data = spark.read.option("header", "false").option("multiline",False).json(input_tweets_folder)
data = data.na.drop()
data = data.withColumn('text_cleaned', regexp_replace('text', '[#|,|&|!|~|*]|http.*', ''))
# ------------------- Grouping individual texts ----------------------#
grouped_df = data.groupby('name').agg(collect_list('text_cleaned').alias("text_aggregated"))
grouped_appended_df = grouped_df.withColumn("text_aggregated_1", concat_ws(". ", "text_aggregated"))
# --------------------------------------------------------------------#
tokenizer = Tokenizer(inputCol="text_aggregated_1", outputCol="words")
wordsData = tokenizer.transform(grouped_appended_df)

#tokenizer = Tokenizer(inputCol="text_cleaned", outputCol="words")
#wordsData = tokenizer.transform(data)
stopwordlist = custom_stop_words()
remover = StopWordsRemover(inputCol="words", outputCol="filtered_col", stopWords=stopwordlist)
filtered = remover.transform(wordsData)
filtered.select("filtered_col").show(truncate=False)

cv = CountVectorizer(inputCol="filtered_col", outputCol="rawFeatures", vocabSize=15000, minDF=5.0)
cvmodel = cv.fit(filtered)
# ------------------------- BOW model ----------------#
vocab = cvmodel.vocabulary # list of words in order, write out for vocab.txt file
d_bow = filtered.count()
w_bow = len(vocab)
vector_udf = udf(lambda vector: vector.numNonzeros(), LongType())
# ----------------------------------------------------#
vocab_broadcast = sc.broadcast(vocab)
featurized = cvmodel.transform(filtered)
nnz_bow = featurized.select(vector_udf('rawFeatures')).groupBy().sum().collect()

sparse_values = udf(lambda v: v.values.tolist(), ArrayType(DoubleType()))
nnz_elements_count = featurized.select(sparse_values('rawFeatures'))
sparse_indices = udf(lambda v: v.indices.tolist(), ArrayType(LongType()))
nnz_elements = featurized.select(sparse_indices('rawFeatures'))
fzipped = f.select('vals','indices').rdd.zipWithIndex().toDF()
fzipped_sep = fzipped.withColumn('vals', fzipped['_1'].getItem("vals"))
fzipped_sep = fzipped_sep.withColumn('indices', fzipped['_1'].getItem("indices"))
fzipped_sep2 = fzipped_sep.select("_2", arrays_zip("indices", "vals"))
nnz_data = fzipped_sep2.select("_2", explode("arrays_zip(indices, vals)"))

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurized)
rescaled = idfModel.transform(featurized)

lda_model = LDA(k=10, maxIter=20)
model = lda_model.fit(rescaled)
modeldf = model.transform(rescaled)
topics = model.describeTopics()
ll = model.logLikelihood(rescaled)
print("Log likelihood is ",ll)
lp = model.logPerplexity(rescaled)
print("Log perplexity is ",lp)

def map_termid_to_word(termindices):
    words = []
    for termid in termindices:
        words.append(vocab_broadcast.value[termid])
        
    return(words)


udf_map_termid_to_word = udf(map_termid_to_word, ArrayType(StringType()))
ldatopics_mapped = topics.withColumn("topic_desc", udf_map_termid_to_word(topics.termIndices))
ldatopics_mapped.select(col("termWeights"), col("topic_desc")).show(truncate=False)


                                     