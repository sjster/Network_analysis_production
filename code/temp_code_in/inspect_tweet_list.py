import pickle
import s3fs
import configparser
import fastparquet as fp
import pickle 
import pandas

def write_s3_files(all_paths_from_s3):
	with open("/home/vt/extra_storage/Production/temporary_files/processed_tweet_files","wb") as f:
		pickle.dump(all_paths_from_s3,f)

def write_s3_files():
	with open("/home/vt/extra_storage/Production/temporary_files/processed_tweet_files","rb") as f:
		pf = pickle.load(f)
	return(pf)

def get_all_processed_files(fs):
	s3_path = "sentimentres/processed/*.parquet"
	all_paths_from_s3 = fs.glob(path=s3_path)
	return(all_paths_from_s3)

def get_all_sentiment(fs):
	sentiment_files = fs.glob(path='sentimentres/results/*.npy')
	for file in sentiment_files:
		print(file)
		print("/home/vt/extra_storage/Production/output/sentiment/" + file.split('/')[2])
		fs.get(file, "/home/vt/extra_storage/Production/output/sentiment/" + file.split('/')[2])

config = configparser.RawConfigParser()
profile_name = 'wasabi'
profile_type = 'wasabi'
config.read('/home/vt/.aws/credentials')
ACCESS_KEY_ID = config.get(profile_name, 'aws_access_key_id')
SECRET_ACCESS_KEY = config.get(profile_name, 'aws_secret_access_key')
if(profile_type == 'wasabi'):
	fs = s3fs.S3FileSystem( key=ACCESS_KEY_ID, secret=SECRET_ACCESS_KEY \
		,client_kwargs={'endpoint_url':'https://s3.wasabisys.com'})

#all_paths_from_s3 = get_all_processed_files(fs)
get_all_sentiment(fs)
