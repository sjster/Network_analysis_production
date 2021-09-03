import pandas as pd
import numpy as np
from scipy.special import softmax
import pickle
import csv
import urllib


with open("/home/vt/extra_storage/Production/temporary_files/processed_tweet_files","rb") as f:
	files = pickle.load(f)

for file in files:
	filename = file.split("/")[2]
	df = pd.read_parquet("/home/vt/extra_storage/Production/output/tweets_tokenized_sentiment.json/" + filename)
	sentiment_file = "/home/vt/extra_storage/Production/output/sentiment/" + filename.split(".")[0] + ".npy"
	sf = np.load(sentiment_file, allow_pickle=True)
	sf1 = np.apply_along_axis(softmax, axis=1, arr=sf)
	print(len(df), len(sf))
	assert(len(df) == len(sf))
	res1, res2, res3, res4 = map(list, zip(*sf1))
	df['anger'] = res1
	df['joy'] = res2
	df['optimism'] = res3
	df['sadness'] = res4
	print(df.head())

