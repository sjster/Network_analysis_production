import numpy as np
from scipy.special import softmax
import csv
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import pandas as pd
import tensorflow as tf
import urllib
import json
import glob
import os

task='emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
PATH = '/home/vt/extra_storage/Production/temporary_files/df_sampled.json/'
PATH = '/home/vt/extra_storage/Production/output/tweets_tokenized_sentiment.json/'
pd.set_option('display.max_colwidth', None)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
     html = f.read().decode('utf-8').split("\n")
     csvreader = csv.reader(html, delimiter='\t')

labels = [row[1] for row in csvreader if len(row) > 1]

model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

def naive_inference():

    for file in glob.glob(PATH + '*.parquet'):
        df = pd.read_parquet(file)

        df['sentiment'] = df['text_tokenized'].apply(lambda x: softmax(model(tf.convert_to_tensor([x]))[0][0].numpy()))
        df3 = pd.DataFrame(df['sentiment'].to_list(), columns=['anger','joy','optimism','sadness'])
        df3['text'] = df['text']
        print(df3[['text', 'anger', 'joy', 'optimism', 'sadness']])
        
        
def inference_1():
    
    filename = '/home/vt/extra_storage/Production/output/tweets_tokenized_sentiment.json/part-00101-3c6f0149-5762-4ffd-8928-d3be02f0d235-c000.snappy.parquet'
    
    for file in glob.glob(PATH + '*.parquet'):
        df = pd.read_parquet(file)
        print(file)
        tokenized = tokenizer(list(df['text']), padding=True, return_tensors='tf')
        res = model.predict(tokenized['input_ids'], batch_size=100, use_multiprocessing=True)
    
inference_1()
    
