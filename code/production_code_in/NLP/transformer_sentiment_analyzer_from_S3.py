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
import s3fs
import configparser
import fastparquet as fp
import pickle

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


# Get a handle for the Wasabi file system
# Wasabi credentials file path has to be provided
def read_wasabi(credentials_file='/home/vt/.aws/credentials'):
    config = configparser.RawConfigParser()
    config.read(credentials_file)

    profile_name = 'wasabi'
    profile_type = 'wasabi'

    ACCESS_KEY_ID = config.get(profile_name, 'aws_access_key_id')
    SECRET_ACCESS_KEY = config.get(profile_name, 'aws_secret_access_key')

    if(profile_type == 'wasabi'):
            fs = s3fs.S3FileSystem( key=ACCESS_KEY_ID, secret=SECRET_ACCESS_KEY \
                    ,client_kwargs={'endpoint_url':'https://s3.wasabisys.com'})
        
    return(fs)


# Read tweets file in parquet format from Wasabi bucket sentiment
# Move processed files to sentimentres/processed/
# Write processed sentiment files in numpy format to sentimentres/results/
def sentiment_on_tweets_parquet_wasabi(s3_path = "sentiment/*.parquet"):

    all_paths_from_s3 = fs.glob(path=s3_path)
    for file in all_paths_from_s3:
      df = pd.read_parquet(fs.open(path=file, mode='rb'))
      print(file)
      tokenized = tokenizer(list(df['text']), padding=True, return_tensors='tf')
      res = model.predict(tokenized['input_ids'], batch_size=100, use_multiprocessing=True)
      output_file = 'sentimentres/results/' + file.split('/')[1].split('.')[0] + '.npy'
      print(output_file)
      with fs.open(output_file, 'wb') as f:
            f.write(pickle.dumps(res['logits'])) 
      fs.mv(file, 'sentimentres/processed/' + file.split('/')[1])
    

read_wasabi()
sentiment_on_tweets_parquet()
    
    