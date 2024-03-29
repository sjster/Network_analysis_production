import pandas as pd
import glob
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData

engine = create_engine("mysql+pymysql://vt:1234@localhost/Tweets")
conn = engine.connect()

# Upload the tweets folder to the database, this overwrites the database, possibly have to do updates instead of
# overwrites as this grows
# The JSON file name is hardcoded, which will be different when the RQ downloader is rerun or the tweet join script is rerun
df = pd.read_json(glob.glob('/home/vt/extra_storage/Production/output/tweets_top5000_joined_by_rank.json/*.json')[0], lines=True)
df.drop(labels='user_id', axis=1).to_sql('tweets', con=engine, if_exists='replace')

