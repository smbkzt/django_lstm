import json

from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, HttpResponse

from . import LSTM

try_lstm = LSTM.TryLstm()


@csrf_exempt
# Create your views here.
def index(request):
    if request.method == 'POST' and request.is_ajax():
        message = request.POST.get("the_origin", "")
        response = request.POST.get("the_comment", "")
        messages = message + " < - > " + response
        answer = try_lstm.predict(messages)
        if answer == "The comment message has disagreement sentiment":
            return HttpResponse(json.dumps(answer),
                                content_type="application/json"
                                )
        else:
            return HttpResponse(json.dumps(answer),
                                content_type="application/json"
                                )
    else: return render(request, "main/index.html")
