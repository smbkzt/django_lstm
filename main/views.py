import json
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, HttpResponse
from django.utils.decorators import method_decorator
from . import LSTM


try_lstm = LSTM.TryLstm()


@method_decorator(csrf_exempt, name='dispatch')
class IndexView(View):
    def get(self, request):
        return render(request, "main/index.html")

    def post(self, request):
        if request.is_ajax():
            message = request.POST.get("the_origin", "")
            response = request.POST.get("the_comment", "")
            messages = message + " < - > " + response
            answer = try_lstm.predict(messages)
            import time
            time.sleep(1.5)
            if answer == "The comment message has disagreement sentiment":
                return HttpResponse(json.dumps(answer),
                                    content_type="application/json"
                                    )
            else:
                return HttpResponse(json.dumps(answer),
                                    content_type="application/json"
                                    )
        else:
            return render(request, "main/index.html")
