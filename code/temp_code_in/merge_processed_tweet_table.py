import pandas as pd
import numpy as np
import pickle

with open("/home/vt/extra_storage/Production/temporary_files/processed_tweet_files","rb") as f:
	files = pickle.load(f)

for file in files:
	filename = file.split("/")[2]
	df = pd.read_parquet("/home/vt/extra_storage/Production/output/tweets_tokenized_sentiment.json/" + filename)
	sentiment_file = "/home/vt/extra_storage/Production/output/sentiment/" + filename.split(".")[0] + ".npy"
	print(df.head())
	sf = np.load(sentiment_file, allow_pickle=True)
	print(sf)
