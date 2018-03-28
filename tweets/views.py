import json

from django.views import View
from django.shortcuts import render, HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .tweepy_code import tweets_load


# Create your views here.
@method_decorator(csrf_exempt, name='dispatch')
class TweetsSearch(View):
    def get(self, request):
        return render(request, "tweets/tweets_search.html")

    def post(self, request):
        if request.is_ajax():
            keywords = request.POST.get("keywords", "")
            if not keywords == "":
                tweeter_account = tweets_load.TwitterAccount()
                keywords = keywords.split(";")
                list_of_tweets = tweeter_account.get_tweets_and_users(keywords)
                nodes_list = []
                edges_list = []
                number = 0
                for dictionary in list_of_tweets:
                    origin_tweet_n = number
                    reply_tweet_n = number+1
                    nodes_list.append({"id": number, "label": dictionary['origin_tweet_user']})
                    nodes_list.append({"id": number+1, "label": dictionary['reply_tweet_user']})
                    number += 2
                    if number > 0:
                        edges_list.append({"from": origin_tweet_n, "to": reply_tweet_n})
                json_response = {
                    "nodes": nodes_list,
                    "edges": edges_list
                }
                return HttpResponse(json.dumps(json_response,
                                               separators=(',', ':')),
                                    content_type='application/json')
