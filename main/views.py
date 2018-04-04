import json

from django.views import View
from django.shortcuts import render, HttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .tf import use_lstm
from .tf import train_and_test as tr
try_lstm = use_lstm.TryLstm()  # LSTM instance creation


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


@method_decorator(csrf_exempt, name='dispatch')
class TrainPageView(View):
    def get(self, request):
        return render(request, "main/retrain_page.html")

    def post(self, request):
        if request.is_ajax():
            numDimensions = request.POST.get("train_dimensions", "")
            maxSeqLength = request.POST.get("train_seqlength", "")
            batchSize = request.POST.get("train_batch", "")
            lstmUnits = request.POST.get("train_units", "")
            numClasses = request.POST.get("train_classes", "")
            training_steps = request.POST.get("train_steps", "")
            cells = request.POST.get("train_cells", "")

            try:
                tr.PrepareData(numDimensions, maxSeqLength, batchSize,
                               lstmUnits, numClasses, training_steps, cells)
                answer = "The training process is done"
            except Exception as e:
                answer = f'Error: {e}'

            return HttpResponse(json.dumps(answer),
                                content_type="application/json"
                                )
        else:
            return render(request, "main/retrain_page.html")
