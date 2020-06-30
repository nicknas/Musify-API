from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from api.agents import *
import spotipy


chatbot_agent = ChatbotAgent("chatbot-musify@404.city", "chatbot-musify")
personal_recommender_agent = PersonalRecommenderAgent("personal-recommender-musify@404.city",
                                                      "personal-recommender-musify")
popular_recommender_agent = PopularRecommenderAgent("popular-recommender-musify@404.city", "popular-recommender-musify")


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

    personal_recommender_agent.start().result()
    chatbot_agent.start().result()
    chatbot_agent.request_save_song_tags.user = user
    chatbot_agent.request_save_song_tags.user_input = request.data['user_input']
    template = Template()
    template.set_metadata("performative", "inform")
    chatbot_agent.add_behaviour(chatbot_agent.request_save_song_tags, template)
    chatbot_agent.request_save_song_tags.join()
    result = json.loads(chatbot_agent.request_save_song_tags.exit_code)

    if result.get('error') is not None:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={"error": result.get('error')})
    else:
        return Response(status=status.HTTP_200_OK, data={'intent': result.get('intent'),
                                                         'response': result.get('response')})


@api_view(['GET'])
def request_recommendations(request, user):
    SPOTIPY_CLIENT_ID = '63cd1c05a2de40b19d4316d23e5271bf'
    SPOTIPY_CLIENT_SECRET = 'e0a096314a2946e4ab0c5a73f9fdd4cd'
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                                                  client_secret=SPOTIPY_CLIENT_SECRET))
    recommendations = {'songs': []}
    genres = []
    artists = []

    distinct_genres = SongTag.objects.filter(user=User.objects.get(user_name=user)).values("genre").distinct()
    distinct_artists = SongTag.objects.filter(user=User.objects.get(user_name=user)).values("artist").distinct()

    # Si no hay más de 2 géneros diferentes o más de 3 artistas diferentes, no tiene sentido realizar deep learning
    if len(distinct_genres) <= 2 or len(distinct_artists) <= 3:
        for genre in distinct_genres[:2]:
            if len(genre['genre']) > 0:
                genres.append(genre['genre'])

        for artist in distinct_artists[:3]:
            artists.append(artist['artist'])

    else:
        popular_recommender_agent.start().result()
        chatbot_agent.request_recommendation.user = user
        template = Template()
        template.set_metadata("performative", "inform")
        chatbot_agent.add_behaviour(chatbot_agent.request_recommendation, template)
        chatbot_agent.request_recommendation.join()
        result = json.loads(chatbot_agent.request_recommendation.exit_code)
        genres = result['genres']
        artists = result['artists']
        chatbot_agent.stop()
        personal_recommender_agent.stop()
        popular_recommender_agent.stop()

    if len(genres) > 0 and len(artists) > 0:
        r = spotify.recommendations(seed_artists=artists, seed_genres=genres, limit=10)
        for song in r['tracks']:
            recommendations['songs'].append({'artist': song['artists'][0]['name'], 'song': song['name']})

        return Response(status=status.HTTP_200_OK, data=recommendations)

    else:
        return Response(status=status.HTTP_400_BAD_REQUEST, data={'error': 'There is no recommendations for this user'})
