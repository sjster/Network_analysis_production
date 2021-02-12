import pandas as pd
import glob
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData

engine = create_engine("mysql+pymysql://vt:1234@localhost/Tweets")
conn = engine.connect()

# Insert all top 5000 pagerank users and their ids, ranked by pagerank
df = pd.read_csv('../data/production_data_in/id_name_by_pagerank.csv')
df.to_sql('page_rank_top5000_users', con=engine, if_exists='replace')

# Insert friends and followers of central figures, along with pagerank, friends and followers count,
# what central figure they are connected to.
df = pd.read_json(glob.glob('/home/vt/extra_storage/Production/data/production_data_in/friends_and_followers_of_central_figures/*.json')[0],lines=True)

df['followers_of_central_figure_string'] = [','.join(map(str, l)) for l in df['followers_of_central_figure']]
df['friends_of_central_figure_string'] = [','.join(map(str, l)) for l in df['friends_of_central_figure']]

df.drop(['friends_of_central_figure','followers_of_central_figure'], axis=1).to_sql('page_rank_top5000_friends_and_followers_of_central_figures', con=engine, if_exists='replace')




