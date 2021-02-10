from pyspark.ml import Pipeline
import pyspark
from pyspark import SparkContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql import functions
from graphframes import *
import os
import argparse
import json
import glob
import pickle
import pandas as pd
from pyspark.sql.functions import *
from graphframes import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pyspark.ml.feature import Bucketizer
import pickle
import logging
import logging.handlers
import os

def setup_logging(logfile, console_handler=True):
    logfile_name = logfile + '.log'
    logfile_object = logfile + '_log'
    rm_logfile = 'rm ' + logfile_name
    if(os.path.exists(logfile_name)):
       os.system(rm_logfile)
    log = logging.getLogger(logfile_object)
    log.setLevel(logging.INFO)
    log.propagate = False
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", logfile_name))
    handler.setFormatter(formatter)
    log.addHandler(handler)

    if(console_handler == True):
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)
        log.addHandler(consoleHandler)
    return(log)

def get_filtered_edges(vertices, edges, g):
    gd = g.degrees
    edges2 = edges.join(gd, gd.id == edges.src).select('src', 'dst', 'relationship', col('degree').alias('src_degree'))
    edges2 = edges2.join(gd, gd.id == edges2.dst).select('src', 'dst', 'relationship', 'src_degree', col('degree').alias('dst_degree'))
    filtered_edges = edges2.filter((col('src_degree') > 1) & (col('dst_degree') > 1))
    
    gd_filter = gd.filter(col('degree') > 1).withColumnRenamed('id','filtered_id')
    filtered_vertices = vertices.join(gd_filter, vertices.id == gd_filter.filtered_id).select('id', 'screen_name')
    
    return(filtered_vertices, filtered_edges)
  
log = setup_logging('../output/graph_metrics_log')

conf = pyspark.SparkConf()
conf.set("spark.ui.showConsoleProgress", "true")
conf.set('spark.local.dir', '/home/vt/extra_storage/')
conf.set('spark.sql.shuffle.partitions', '200')
conf.set('spark.driver.cores','18'),
conf.set('spark.executor.cores','18'),
conf.set('spark.sql.execution.arrow.enabled','true')
conf.set('spark.task.cpus','18')
conf.set('spark.sql.catalogImplementation', 'hive')
conf.set('spark.serializer.objectStreamReset', '100')
#SparkContext.setLogLevel(logLevel="WARN")
SparkContext.setSystemProperty('spark.executor.memory', '48g')
SparkContext.setSystemProperty('spark.driver.memory', '48g')
sc = SparkContext(appName='mm_exp', conf=conf)
sc.setLogLevel("INFO")
sc.setCheckpointDir('/home/vt/extra_storage/spark_checkpoint')
spark = SparkSession(sc)

sqlcontext = pyspark.sql.SQLContext(sc)

print(spark.sparkContext.getConf().getAll())

log.info('Reading from ../data')

edges = spark.read.parquet('../data/edges')
vertices = spark.read.parquet('../data/vertices')
g = GraphFrame(vertices, edges)
fv, fe = get_filtered_edges(vertices, edges, g)

nvert = g.vertices.count()
nedges = g.edges.count()
maxd = nvert*(nvert - 1) / 2
# This is the upper bound since bidirectional edges are counted separately
# Sparse matrix, since this value id close to e-8
sparsity = nedges / maxd 
print("Sparsity is ", sparsity)

log.info('Number of vertices -- {:d}'.format(nvert))
log.info('Number of edges -- {:d}'.format(nedges))
log.info('Sparsity -- {:10f}'.format(sparsity))

# Random and regular networks have homogeneous degree distributions
# Small-world network have degree distributions between these two
# Scale-free have highly heterogeneous degree distributions
gindegrees = g.inDegrees
goutdegrees = g.outDegrees
gdegrees = g.degrees

hist_dict = {}
hist_dict['sparsity'] = sparsity

x = [ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 5000, 10000, 25000, 50000, 75000, 100000, 200000, 500000, 1000000, float('Inf') ]
# Total degree distribution
bucketizer = Bucketizer(splits= x,inputCol="degree", outputCol="buckets")
df_buck = bucketizer.setHandleInvalid("keep").transform(gdegrees).groupby(col('buckets')).count()
df = df_buck.sort(col('buckets')).toPandas()
df['x'] = x[:-1]
hist_dict['total'] = df

# Indegree distribution
bucketizerin = Bucketizer(splits=x,inputCol="inDegree", outputCol="buckets")
df_buck = bucketizerin.setHandleInvalid("keep").transform(gindegrees).groupby(col('buckets')).count()
df = df_buck.sort(col('buckets')).toPandas()
df['x'] = x[:-1]
hist_dict['in'] = df

# Outdegree distribution
bucketizerout = Bucketizer(splits=x,inputCol="outDegree", outputCol="buckets")
df_buck = bucketizerout.setHandleInvalid("keep").transform(goutdegrees).groupby(col('buckets')).count()
df = df_buck.sort(col('buckets')).toPandas()
df['x'] = x[:-1]
hist_dict['out'] = df

with open('../output/hist_dict.pkl','wb') as f:
    pickle.dump(hist_dict,f)


sc.stop()
