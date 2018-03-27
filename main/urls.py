from django.urls import path
from . import views as main_views

urlpatterns = [
    path('', main_views.home, name="home_page"),
    path('try-lstm', main_views.TryLstm.as_view(), name="try_lstm_page"),
]
