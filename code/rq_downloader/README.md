
1. Run this in the environment rqenv

2. Run this from the folder /home/vt/RQ - where the RQ service is set up to monitor for jobs


The file job_submission.py pulls tweets corresponding to hashtags or userids in the file passed through the --file flag. The output folder is where the downloaded tweets in JSON files are located. This is passed using the --output flag. Additionally, this folder gets synced to Google Drive 'Twitter_rsettlag' (Need to parameterize this)

1. To download tweets based off hashtags
python job_submission.py --file hashtags.in --output /home/vt/extra_storage/twitter_data 

2. To download tweets based off a userid
python job_submission.py --file input_userids.in --output /home/vt/extra_storage/twitter_data --function download_tweets
