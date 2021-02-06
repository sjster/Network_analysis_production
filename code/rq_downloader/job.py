import sys
import shutil
import calendar
import time
import datetime
sys.path.append('/home/mapped/liberals_scripts')
import credentials
from credentials import *    # This will allow us to use the keys as variables
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing
import jsonpickle
import os
import logging_mod


def read_hashtags(input_file):
    f = open(input_file,'r')
    lines = f.readlines()
    terms = [term.strip() for term in lines]
    f.close()
    return(terms)


def read_user_ids(input_file):
    f = open(input_file,'r')
    lines = f.readlines()
    terms = [term.strip() for term in lines]
    f.close()
    return(terms)


def job_func(input_hashtags_file, output_folder):
    
  log = logging_mod.setup_logging('Jobs.', console_handler=False)

  log.info("Output folder {:s}".format(output_folder))

  #terms = ['nevertrump', 'loveoverhate', 'resist']
  terms = read_hashtags(input_hashtags_file)

  for searchQuery in terms:
    maxTweets = 100000 # Some arbitrary large number
    tweetsPerQry = 100  # this is the max the API permits
    ts = calendar.timegm(time.gmtime())
    fName = searchQuery[0:] + '.' + str(ts) + '.txt' # We'll store the tweets in a text file.

    log.info("Getting data {:s}".format(searchQuery))

    # If results from a specific ID onwards are reqd, set since_id to that ID.
    # else default to no lower limit, go as far back as API allows
    sinceId = None

    # If results only below a specific ID are, set max_id to that ID.
    # else default to no upper limit, start from the most recent tweet matching the search query.
    max_id = -1

    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

    tweetCount = 0
    print("Downloading max {0} tweets".format(maxTweets))
    with open(fName, 'w') as f:
        while tweetCount < maxTweets:
            
            if(tweetCount % 20000):
                log.info("Downloading tweet {:d}".format(tweetCount))
                
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode='extended')
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                since_id=sinceId, tweet_mode='extended')
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                max_id=str(max_id - 1), tweet_mode='extended')
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                max_id=str(max_id - 1),
                                                since_id=sinceId, tweet_mode='extended')
                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    f.write(jsonpickle.encode(tweet._json, unpicklable=False) +
                            '\n')
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                log.exception(str(e))
                break

    print("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
    log.info("Downloaded {:d} tweets".format(tweetCount))
    source = fName
    dest = output_folder + source
    shutil.move(source,dest)

  rclone_command = 'rclone sync ' + output_folder + ' remote_google:Twitter_rsettlag -P'
  log.info("Rclone folder {:s}".format(rclone_command))
  os.system(rclone_command)


def download_tweets(input_userid_file='input_userids.in', output_folder='/home/vt/extra_storage/twitter_data/dissertation_tweets/'):

  log = logging_mod.setup_logging('Jobs', console_handler=False)

  log.info("Output folder {:s}".format(output_folder))

  #terms = ['nevertrump', 'loveoverhate', 'resist']
  terms = read_user_ids(input_userid_file)
  log.info("Found {0} user ids to download".format(len(terms)))

  for searchQuery in terms:
    
        ts = calendar.timegm(time.gmtime())
        fName = searchQuery[0:] + '.' + str(ts) + '.txt' # We'll store the tweets in a text file.
        
        log.info("Getting data for {:s}".format(searchQuery))

        # Authentication and access using keys:
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        tweetCount = 0
        with open(fName, 'w') as f:
            try:
                tweets = api.user_timeline(user_id=searchQuery, count=1000, tweet_mode="extended")
                for tweet in tweets:
                        f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
                tweetCount += len(tweets)
            except Exception:
                log.info("Tweepy error for ", searchQuery)
                log.exception("Exception in Tweepy")
            
        source = fName
        dest = output_folder + source
        log.info("Downloaded {0} tweets, Saved to {1}".format(tweetCount, dest))
        shutil.move(source,dest)

  log.info("Job done")


download_tweets()
