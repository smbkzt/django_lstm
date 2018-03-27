import json

from django.views import View
from django.shortcuts import render, HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
@method_decorator(csrf_exempt, name='dispatch')
class TweetsSearch(View):
    def get(self, request):
        return render(request, "tweets/tweets_search.html")

    def post(self, request):
        if request.is_ajax():
            keywords = request.POST.get("keywords", "")
            if not keywords == "":
                keywords = keywords.split(";")
                return HttpResponse(json.dumps(keywords),
                                    content_type="application/json"
                                    )
