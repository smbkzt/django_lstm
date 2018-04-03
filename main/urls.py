from django.urls import path
from . import views as main_views
from django.contrib.auth.decorators import login_required

urlpatterns = [
    path('', main_views.home, name="home_page"),
    path('try-lstm', main_views.TryLstmView.as_view(), name="try_lstm_page"),
    path('train-lstm', login_required(main_views.TrainPageView.as_view()),
         name="train_page"),
]
