from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from api.models import User
from rest_framework import status

@api_view(['POST'])
def login(request):
    user_to_log = User(**request.data)
    try:
        user = User.objects.get(user_name=user_to_log.user_name, password=user_to_log.password)
        return Response(status=status.HTTP_200_OK, data={})
    except User.DoesNotExist:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'Invalid user/password'})

@api_view(['POST'])
def register(request):
    user_to_log = User(**request.data)
    try:
        user = User.objects.get(user_name=user_to_log.user_name)
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'User exists in the app'})
    except User.DoesNotExist:
        user_to_log.save()
        return Response(status=status.HTTP_200_OK, data={})