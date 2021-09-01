import numpy as np
from scipy.special import softmax
import csv
from transformers import *
import urllib
import json
import os

from pyspark.sql.functions import udf, pandas_udf


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task='emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
     html = f.read().decode('utf-8').split("\n")
     csvreader = csv.reader(html, delimiter='\t')

labels = [row[1] for row in csvreader if len(row) > 1]

model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

    
#@pandas_udf(StringType, PandasUDFType.SCALAR_ITER)
def get_results(inp):
        tokenized = tokenizer(seq, padding=True, return_tensors='tf')['input_ids']
        model = TFAutoModelForSequenceClassification()
        model.set_weights(modelbc.value)
        output = model(tokenized)
        scores = output[0][0].numpy()
        return(scores)


def spark_tokenize():
    
    def tokenize(seq):
        return tokenizer(seq, padding=True)['input_ids']
    
    input_tweets_folder = "/home/vt/extra_storage/Production/output/tweets_top5000_joined_by_rank.json"
    df = spark.read.option("header", "false").option("multiline",False).json(input_tweets_folder)
    tokenize_udf = udf(tokenize, ArrayType(IntegerType()))
    df = df.dropna(how='any', subset='text')
    df = df.dropDuplicates(['tweet_id'])
    df = df.withColumn("text_tokenized", tokenize_udf("text"))
    df.write.option('format','json').mode('overwrite').save('/home/vt/extra_storage/Production/output/tweets_tokenized_sentiment.json')

def hello():

    text = "Good night 😊"
    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    
    print(dir(model))
    
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
         l = labels[ranking[i]]
         s = scores[ranking[i]]
         print(f"{i+1}) {l} {np.round(float(s), 4)}")
            
def classify_text():
    
    text = ["Welcome to the club!", "Get out of here"]
    textlength = len(text)
    encoded_input = tokenizer(text, padding= True, return_tensors='tf')
    
    output = model(encoded_input)
    
    for i in range(0, textlength):
        score = softmax(output[0][i].numpy()) 
        print(list(zip(labels, score)))
    

spark_tokenize()