import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData

engine = create_engine("mysql+pymysql://vt:1234@localhost/Tweets")
conn = engine.connect()

df = pd.read_csv('../data/production_data_in/id_name_by_pagerank.csv')
df.to_sql('users', con=engine, if_exists='replace')


