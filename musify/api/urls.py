from django.urls import path, include
from api import views

urlpatterns = [
    path('login/', views.login),
    path('register/', views.register)
]