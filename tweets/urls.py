from django.urls import path
from . import views as tweet_views

urlpatterns = [
    path('', tweet_views.TweetsSearch.as_view(), name="tweet_page"),
]
