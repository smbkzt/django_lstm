#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3.6
# # -*- coding: UTF-8 -*-

import re
import time
import logging
import string

import tweepy
from tweepy import OAuthHandler

from . import config


class TwitterAccount():
    def get_api(self):
        try:
            self.consumer_key = config.consumer_key
            self.consumer_secret = config.consumer_secret
            self.access_token = config.access_token
            self.access_secret = config.access_secret

            self.auth = OAuthHandler(self.consumer_key, self.consumer_secret)
            self.auth.set_access_token(self.access_token, self.access_secret)
            self.api = tweepy.API(self.auth, wait_on_rate_limit_notify=True)
        except tweepy.TweepError as exception:
            logging.exception(exception)
        return self.api

    def get_tweets_and_users(self, keywords):
        self.api = self.get_api()
        list_of_tweets = []
        for tweet in tweepy.Cursor(self.api.search,
                                   q=keywords,
                                   rpp=100,
                                   result_type="recent",
                                   include_entities=True,
                                   lang="en").items(100):
            if tweet.in_reply_to_status_id:
                try:
                    tweet_dict = {}
                    origin_tweet = self.api.get_status(
                                            id=tweet.in_reply_to_status_id
                                            )
                    if not origin_tweet.user.id == tweet.user.id:
                        tweet_dict['origin_tweet_user'] = origin_tweet.user.screen_name
                        tweet_dict['origin_tweet_user_image'] = origin_tweet.user.profile_image_url
                        tweet_dict['origin_tweet_title'] = origin_tweet.text

                        tweet_dict['reply_tweet_user'] = tweet.user.screen_name
                        tweet_dict['reply_tweet_user_image'] = tweet.user.profile_image_url
                        tweet_dict['reply_tweet_title'] = tweet.text
                        list_of_tweets.append(tweet_dict)
                except tweepy.TweepError as error:
                    print(error)
        return list_of_tweets
