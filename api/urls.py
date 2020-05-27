from django.urls import path, include
from api import views

urlpatterns = [
    path('login', views.login),
    path('register', views.register),
    path('<str:user>/request_songs', views.request_songs),
    path('<str:user>/request_recommendations', views.request_recommendations)
]