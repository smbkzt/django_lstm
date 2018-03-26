from django.urls import path
from .views import TryLstm
from .views import home

urlpatterns = [
    path('', home, name="home_page"),
    path('try-lstm', TryLstm.as_view(), name="try_lstm_page")
]
