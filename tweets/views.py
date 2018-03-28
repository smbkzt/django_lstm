import json

from django.views import View
from django.shortcuts import render, HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .tweepy_code import tweets_load
from main.views import try_lstm


# Create your views here.
@method_decorator(csrf_exempt, name='dispatch')
class TweetsSearch(View):
    def get(self, request):
        return render(request, "tweets/tweets_search.html")

    def post(self, request):
        # TODO: make the box show original and response messages
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
                    message = dictionary["origin_tweet_title"] + " < - > " + \
                              dictionary["reply_tweet_title"]
                    prediction = try_lstm.predict(message)
                    edge_label = "disagreed" if "disagreement" in prediction else "agreed"

                    origin_tweet_n = number
                    reply_tweet_n = number+1
                    dic1 = {"id": number,
                            "label": dictionary['origin_tweet_user'],
                            "image": dictionary['origin_tweet_user_image'],
                            "title": dictionary["origin_tweet_title"]}
                    nodes_list.append(dic1)

                    dic2 = {"id": number+1,
                            "label": dictionary['reply_tweet_user'],
                            "image": dictionary['reply_tweet_user_image'],
                            "title": dictionary["reply_tweet_title"]}
                    nodes_list.append(dic2)
                    number += 2
                    if number > 0:
                        dic3 = {"from": origin_tweet_n,
                                "to": reply_tweet_n,
                                "label": edge_label}
                        edges_list.append(dic3)
                json_response = {
                    "nodes": nodes_list,
                    "edges": edges_list
                }
                return HttpResponse(json.dumps(json_response,
                                               separators=(',', ':')),
                                    content_type='application/json')
