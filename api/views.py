from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from api.models import User
from api.models import SongTag
from rest_framework import status
from google.api_core.exceptions import InvalidArgument
from spotipy.oauth2 import SpotifyClientCredentials
from api.agents import ChatbotAgent
import spotipy
import dialogflow_v2


@api_view(['POST'])
def login(request):
    user_to_log = User(**request.data)
    try:
        user = User.objects.get(user_name=user_to_log.user_name, password=user_to_log.password)
        #dummy = ChatbotAgent("chatbot-musify@404.city", "chatbot-musify")
        #dummy.start()
        SongTag.objects.filter(user = user).delete()
        return Response(status=status.HTTP_200_OK, data={})
    except User.DoesNotExist:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'Invalid user/password'})


@api_view(['POST'])
def register(request):
    user_to_log = User(**request.data)
    try:
        user = User.objects.get(user_name=user_to_log.user_name)
        #chatbot_agent = ChatbotAgent("chatbot-musify@404.city", "chatbot-musify")
        #chatbot_agent.start()
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'User exists in the app'})
    except User.DoesNotExist:
        user_to_log.save()
        return Response(status=status.HTTP_200_OK, data={})


@api_view(['POST'])
def request_songs(request, user):
    client = dialogflow_v2.SessionsClient.from_service_account_json(
        'dialog_flow_credentials/musifychatbot-qkmsfp-64a945d16557.json')
    session = client.session_path('musifychatbot-qkmsfp', user + '-musify_api')
    text_input = dialogflow_v2.types.TextInput(text=request.data['user_input'], language_code='es-ES')
    query_input = dialogflow_v2.types.QueryInput(text=text_input)
    SPOTIPY_CLIENT_ID = '63cd1c05a2de40b19d4316d23e5271bf'
    SPOTIPY_CLIENT_SECRET = 'e0a096314a2946e4ab0c5a73f9fdd4cd'
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                                                  client_secret=SPOTIPY_CLIENT_SECRET))
    try:
        response_chatbot = client.detect_intent(session=session, query_input=query_input)
        if response_chatbot.query_result.intent.display_name == 'Recoger canción':
            results = spotify.search(q='track:' + request.data['user_input'], type='track', limit=10)
            if len(results['tracks']['items']) == 0:
                response_chatbot.query_result.fulfillment_text = "No he podido encontrar canciones para la canción que has puesto"
            for track in results['tracks']['items']:
                response_chatbot.query_result.fulfillment_text += '\n -' + track['name'] + ' del álbum ' + \
                                                                  track['album']['name'] + ' y artista ' + \
                                                                  track['artists'][0]['name']
                artist_full = spotify.artist(track['artists'][0]['uri'])
                available_genres = spotify.recommendation_genre_seeds()['genres']
                genres = [genre.replace(" ", "-") for genre in artist_full['genres'] if genre.replace(" ", "-") in available_genres]
                song_tag = SongTag(song = track['id'], artist = track['artists'][0]['id'], genre = genres[0] if len(genres) > 0 else '', release_year = track['album']['release_date'], user = User.objects.get(user_name=user))
                song_tag.save()

        if response_chatbot.query_result.intent.display_name == 'Recoger artista':
            results = spotify.search(q='artist:' + request.data['user_input'], type='artist')
            if len(results['artists']['items']) == 0:
                response_chatbot.query_result.fulfillment_text = "No he podido encontrar canciones para el artista que has puesto"

            else:
                artist = results['artists']['items'][0]
                for album in spotify.artist_albums(artist['uri'], album_type='album', limit=1)['items']:
                    for track in spotify.album_tracks(album['uri'], limit=10)['items']:
                        response_chatbot.query_result.fulfillment_text += '\n -' + track['name'] + ' del álbum ' + \
                                                                          album['name'] + ' y artista ' + artist['name']
                        artist_full = spotify.artist(artist['uri'])
                        print(artist_full['genres'])
                        available_genres = spotify.recommendation_genre_seeds()['genres']
                        genres = [genre.replace(" ", "-") for genre in artist_full['genres'] if genre.replace(" ", "-") in available_genres]
                        song_tag = SongTag(song = track['id'], artist = artist['id'], genre = genres[0] if len(genres) > 0 else '', release_year = album['release_date'], user = User.objects.get(user_name=user))
                        song_tag.save()

        if response_chatbot.query_result.intent.display_name == 'Recoger album':
            results = spotify.search(q='album:' + request.data['user_input'], type='album', limit=1)
            if len(results['albums']['items']) == 0:
                response_chatbot.query_result.fulfillment_text = "No he podido encontrar canciones para el álbum que has puesto"
            else:
                for album in results['albums']['items']:
                    for track in spotify.album_tracks(album['uri'], limit=10)['items']:
                        response_chatbot.query_result.fulfillment_text += '\n -' + track['name'] + ' del álbum ' + album[
                            'name'] + ' y artista ' + album['artists'][0]['name']
                        artist_full = spotify.artist(album['artists'][0]['uri'])
                        available_genres = spotify.recommendation_genre_seeds()['genres']
                        genres = [genre.replace(" ", "-") for genre in artist_full['genres'] if genre.replace(" ", "-") in available_genres]
                        song_tag = SongTag(song = track['id'], artist = album['artists'][0]['id'], genre = genres[0] if len(genres) > 0 else '', release_year = album['release_date'], user = User.objects.get(user_name=user))
                        song_tag.save()

        return Response(status=status.HTTP_200_OK, data={'intent': response_chatbot.query_result.intent.display_name,
                                                         'response': response_chatbot.query_result.fulfillment_text})

    except InvalidArgument:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={"error": "Unrecognized exception"})

@api_view(['GET'])
def request_recommendations(request, user):
    SPOTIPY_CLIENT_ID = '63cd1c05a2de40b19d4316d23e5271bf'
    SPOTIPY_CLIENT_SECRET = 'e0a096314a2946e4ab0c5a73f9fdd4cd'
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                                                  client_secret=SPOTIPY_CLIENT_SECRET))
    recommendations = {'songs': []}
    genres = []
    artists = []

    for genre in SongTag.objects.filter(user = User.objects.get(user_name=user)).values("genre").distinct()[:2]:
        if len(genre['genre']) > 0:
            genres.append(genre['genre'])

    for artist in SongTag.objects.filter(user = User.objects.get(user_name=user)).values("artist").distinct()[:3]:
        artists.append(artist['artist'])

    print(genres)
    print(artists)
    if len(genres) > 0 and len(artists) > 0:
        r = spotify.recommendations(seed_artists=artists, seed_genres=genres, limit=10)
        for song in r['tracks']:
            recommendations['songs'].append({'artist': song['artists'][0]['name'], 'song': song['name']})
        return Response(status=status.HTTP_200_OK, data=recommendations)

    else:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'There is no recommendations for this user'})
