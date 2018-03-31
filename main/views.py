import json

from django.views import View
from django.shortcuts import render, HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .LSTM import TryLstm

try_lstm = TryLstm()  # LSTM instance creation


@method_decorator(csrf_exempt, name='dispatch')
class TryLstmView(View):
    def get(self, request):
        return render(request, "main/try_lstm.html")

    def post(self, request):
        if request.is_ajax():
            message = request.POST.get("the_origin", "")
            response = request.POST.get("the_comment", "")
            if any(i == "" for i in [message, response]):
                return render(request, "main/try_lstm.html")
            messages = message + " < - > " + response
            answer = try_lstm.predict(messages)
            return HttpResponse(json.dumps(answer),
                                content_type="application/json"
                                )
        else:
            return render(request, "main/try_lstm.html")


def home(request):
    return render(request, "main/home_page.html")
