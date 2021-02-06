from redis import Redis
from rq import Queue
from rq_scheduler import Scheduler
from datetime import datetime
from job import job_func, download_tweets
from datetime import timedelta
import argparse

if __name__ ==  "__main__":
    
    parser = argparse.ArgumentParser(description='Job submission for Python RQ')
    parser.add_argument('--filename', type=str, default='hashtags.in',
                    help='The file that has the hashtags on separate lines or userids on separate lines')
    parser.add_argument('--output', type=str, default='/home/vt/extra_storage/twitter_data/dissertation_tweets2/',
                    help='The output folder to write the downloaded tweets')
    parser.add_argument('--function', type=str, default='job_func', help='Function to call, \n 1) job_func for downloading tweets based off hashtags \n 2) download_tweets to download tweets based off user ids')
    
    args = parser.parse_args()
    print(args)
    input_hashtags_file = args.filename
    output_folder = args.output
    function = args.function
    
    scheduler = Scheduler(connection=Redis()) # Get a scheduler for the "default" queue
    if(function == 'job_func'):
        job = scheduler.schedule(scheduled_time=datetime.utcnow(), func=job_func, args=[input_hashtags_file, output_folder], timeout=18000, repeat=1, interval=3600)
        
    elif(function == 'download_tweets'):
        job = scheduler.schedule(scheduled_time=datetime.utcnow(), func=download_tweets, args=[input_hashtags_file, output_folder], timeout=18000, repeat=5, interval=900)
       
    print("Enqueued job ",job)
