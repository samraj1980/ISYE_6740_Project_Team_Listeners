import tweepy
import sqlite3
import json
from requests.exceptions import Timeout, ConnectionError
import time
from time import sleep
import configparser
import ssl
import pandas as pd
import pyodbc

# Loading the config file at run time
config = configparser.ConfigParser()
config.read('config.ini')

# Add your Twitter API credentials
consumer_key = config.get('Twitter_Settings' , 'consumer_key')
consumer_secret = config.get('Twitter_Settings' , 'consumer_secret')
access_key =  config.get('Twitter_Settings' ,'access_key')
access_secret = config.get('Twitter_Settings' ,'access_secret')


# Handling authentication with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

# Create a wrapper for the API provided by Twitter
api = tweepy.API(auth)


server = 'tcp:isye-6420-project.database.windows.net,1433' 
database = 'topic_modelling' 
username = 'project_administrator' 
password = 'isye_6420_admin' 
conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)


#SQLite cursor
c = conn.cursor()


def read_csv():
    driver = pd.read_csv("115th-Congress-House-seeds.csv", header=None)
    
    return(driver)
    


# We need to implement StreamListener to use Tweepy to listen to Twitter
def on_status(user):

        try:  
            for tweet in tweepy.Cursor(api.user_timeline, screen_name=user, exclude_replies=True,  tweet_mode='extended',  count = 10).items():  
                tweet_text = tweet.full_text
                time = tweet.created_at  
                tweeter = tweet.user.screen_name  
                
                
                tweet_dict = {"tweet_text" : tweet_text.strip(), "timestamp" : str(time), "user" :tweeter}  
                tweet_json = json.dumps(tweet_dict) 
                
                # Exclude retweets, too many mentions and too many hashtags
                if not any(('RT @' in tweet_text.strip(), 'RT' in tweet_text.strip() )): 
                    
                    # Filter tweets only in the last 2 years
                    if ((str(time)[:4] == '2019') or  (str(time)[:4] == '2020' )): 
                        #Insert into SQLite
                        c.execute("INSERT INTO tweets_raw ([tweet_text], [time], [User]) VALUES ( ?, ?, ?)", (tweet_text.strip(), str(time), tweeter))

                #Commit DB transactions
                conn.commit()
                            
        except tweepy.TweepError:
            sleep(60)
            
if __name__ == '__main__':
    
    driver = read_csv()
    for index in range(len(driver)):
            user = driver.iloc[index,0]
            print(index, user)
            on_status(user)
