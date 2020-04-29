from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from api.models import User
from rest_framework import status
from google.api_core.exceptions import InvalidArgument
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import dialogflow_v2

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

@api_view(['POST'])
def request_songs(request, user):
    client = dialogflow_v2.SessionsClient.from_service_account_json('dialog_flow_credentials/musifychatbot-qkmsfp-64a945d16557.json')
    session = client.session_path('musifychatbot-qkmsfp', user + '-musify_api')
    text_input = dialogflow_v2.types.TextInput(text=request.data['user_input'], language_code='es-ES')
    query_input = dialogflow_v2.types.QueryInput(text=text_input)
    SPOTIPY_CLIENT_ID = '63cd1c05a2de40b19d4316d23e5271bf'
    SPOTIPY_CLIENT_SECRET = 'e0a096314a2946e4ab0c5a73f9fdd4cd'
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id = SPOTIPY_CLIENT_ID, client_secret = SPOTIPY_CLIENT_SECRET))
    try:
        response_chatbot = client.detect_intent(session=session, query_input=query_input)
        if (response_chatbot.query_result.intent.display_name == 'Recoger canción'):
            results = spotify.search(q='track:' + request.data['user_input'], type='track')
            if len(results['tracks']['items']) == 0:
                response_chatbot.query_result.fulfillment_text = "No he podido encontrar canciones para la canción que has puesto"
            for track in results['tracks']['items']:
                response_chatbot.query_result.fulfillment_text += '\n -' + track['name'] + ' del álbum ' + track['album']['name'] + ' y artista ' + track['artists'][0]['name']

        
        if (response_chatbot.query_result.intent.display_name == 'Recoger artista'):
            results = spotify.search(q='artist:' + request.data['user_input'], type='artist')
            if len(results['artists']['items']) == 0:
                response_chatbot.query_result.fulfillment_text = "No he podido encontrar canciones para el artista que has puesto"
            
            else:
                artist = results['artists']['items'][0]
                for album in spotify.artist_albums(artist['uri'], album_type='album')['items']:
                    for track in spotify.album_tracks(album['uri'])['items']:
                        response_chatbot.query_result.fulfillment_text += '\n -' + track['name'] + ' del álbum ' + album['name'] + ' y artista ' + artist['name']
        
        if (response_chatbot.query_result.intent.display_name == 'Recoger album'):
            results = spotify.search(q='album:' + request.data['user_input'], type='album')
            if len(results['albums']['items']) == 0:
                response_chatbot.query_result.fulfillment_text = "No he podido encontrar canciones para el álbum que has puesto"
            for album in results['albums']['items']:
                for track in spotify.album_tracks(album['uri'])['items']:
                    response_chatbot.query_result.fulfillment_text += '\n -' + track['name'] + ' del álbum ' + album['name'] + ' y artista ' + album['artists'][0]['name']

        return Response(status=status.HTTP_200_OK, data={'intent': response_chatbot.query_result.intent.display_name, 'response': response_chatbot.query_result.fulfillment_text})

    except InvalidArgument:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={"error": "Unrecognized exception"})
