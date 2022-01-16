import tomotopy as  tp
import nltk
from nltk.corpus import stopwords
import pandas as pd
import glob

stop_words = set(stopwords.words('english'))

f = glob.glob('/home/vt/extra_storage/Production/output/tweets_LanaLokteff.txt/*.json')
df = pd.read_json(f[0], lines=True)
print(df.head())

test = pd.Series(df['text'].apply(lambda x: x.split('.')).values[0])
test_cleaned = test.apply(lambda x: x.strip().split())
test_cleaned = test_cleaned.apply(lambda x: [w for w in x if not w.lower() in stop_words])
print(test_cleaned)

def get_MGLDA(test_cleaned):
    mdl = tp.MGLDAModel(k_g=5, k_l=5, min_cf=1, rm_top=10)
    for elem in test_cleaned:
        mdl.add_doc(elem)

    for i in range(0,100,10):
        mdl.train(20)
        print(i, mdl.ll_per_word)

    for k in range(mdl.k_g):
        print(mdl.get_topic_words(k, top_n=10))

    for k in range(mdl.k_l):
        print(mdl.get_topic_words(mdl.k_g + k, top_n=10))
        
def get_HLDA(test_cleaned):
    mdl = tp.HLDAModel(depth=2, min_cf=1, rm_top=10)
    for elem in test_cleaned:
        if(elem):
            mdl.add_doc(elem)
    
    print("Added docs")
    
    for i in range(0,100,10):
        mdl.train(20)
        print(i, mdl.ll_per_word)
        
    for k in range(mdl.k):
        if not mdl.is_live_topic(k):
            continue
        print( mdl.parent_topic(k), mdl.level(k), mdl.num_docs_of_topic(k) )
        print('Top 10 words of global topic #{}'.format(k))
        print(mdl.get_topic_words(k, top_n=10))
        
        
get_MGLDA(test_cleaned)





