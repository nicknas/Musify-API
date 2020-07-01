import dialogflow_v2
import numpy
import pandas
import random
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from django.db.models import Count
from sklearn import preprocessing
from spade import agent
from spade.behaviour import *
import json
import tensorflow as tf

from google.api_core.exceptions import InvalidArgument
from api.models import SongTag, User
from channels.db import database_sync_to_async


class ChatbotAgent(agent.Agent):
    class SendPersonalRecommenderRequest(OneShotBehaviour):
        async def run(self):
            msg = Message(to="personal-recommender-musify@404.city")
            msg.set_metadata("performative", "query")
            msg.body = self.user

            await self.send(msg)
            print("Requested recommendation to personal recommender")
            recommendations = await self.receive(timeout=90)
            if recommendations is not None:
                self.exit_code = recommendations.body
            else:
                self.exit_code = "No se han podido generar recomendaciones"

    class SendSongTagsSaveRequest(OneShotBehaviour):
        async def run(self):
            client = dialogflow_v2.SessionsClient.from_service_account_json(
                'dialog_flow_credentials/musifychatbot-qkmsfp-64a945d16557.json')
            session = client.session_path('musifychatbot-qkmsfp', self.user + '-musify_api')
            text_input = dialogflow_v2.types.TextInput(text=self.user_input, language_code='es-ES')
            query_input = dialogflow_v2.types.QueryInput(text=text_input)

            try:
                response_chatbot = client.detect_intent(session=session, query_input=query_input)
                input = self.user_input
                msg = Message(to="personal-recommender-musify@404.city")
                msg.set_metadata("performative", "query")
                if response_chatbot.query_result.intent.display_name == 'Recoger cancion v2':
                    input = response_chatbot.query_result.parameters.fields["song"].string_value
                if response_chatbot.query_result.intent.display_name == 'Recoger album v2':
                    input = response_chatbot.query_result.parameters.fields["disco"].string_value
                    print(input)
                if response_chatbot.query_result.intent.display_name == 'Recoger artista v2':
                    input = response_chatbot.query_result.parameters.fields["cantante"].string_value

                response = response_chatbot.query_result.fulfillment_text
                msg.body = json.dumps({'user': self.user,
                                       'user_input': input,
                                       'intent': response_chatbot.query_result.intent.display_name,
                                       'response': response})
                await self.send(msg)
                response_save_tags = await self.receive(timeout=90)
                self.exit_code = json.dumps({"intent": response_chatbot.query_result.intent.display_name,
                                             "response": response_save_tags.body})
            except InvalidArgument:
                self.exit_code = json.dumps({"error": "Unrecognized exception"})

    async def setup(self):
        print("Chatbot agent started")
        self.request_save_song_tags = self.SendSongTagsSaveRequest()
        self.request_recommendation = self.SendPersonalRecommenderRequest()


class PersonalRecommenderAgent(agent.Agent):
    class ReceiveSaveSongTagsRequest(OneShotBehaviour):

        @database_sync_to_async
        def save_song_tag(self, song, artist, genre, release_year, user):
            song_tag = SongTag(song=song, artist=artist, genre=genre, release_year=release_year,
                               user=User.objects.get(user_name=user))
            song_tag.save()

        async def run(self):
            song_tags_request_message = await self.receive(timeout=90)
            song_tags_request = json.loads(song_tags_request_message.body)
            user = song_tags_request['user']
            user_input = song_tags_request['user_input']
            intent = song_tags_request['intent']
            response = song_tags_request['response']

            if intent == 'Recoger canción' or intent == 'Recoger cancion v2':
                results = self.agent.spotify_api.search(q='track:' + user_input, type='track', limit=10)
                if len(results['tracks']['items']) == 0:
                    response = "No he podido encontrar canciones para la canción que has puesto"
                for track in results['tracks']['items']:
                    response += '\n -' + track['name'] + ' del álbum ' + \
                                track['album']['name'] + ' y artista ' + \
                                track['artists'][0]['name']
                    artist_full = self.agent.spotify_api.artist(track['artists'][0]['uri'])
                    available_genres = self.agent.spotify_api.recommendation_genre_seeds()['genres']
                    genres = [genre.replace(" ", "-") for genre in artist_full['genres'] if
                              genre.replace(" ", "-") in available_genres]

                    if len(genres) > 0:
                        await self.save_song_tag(song=track['id'], artist=track['artists'][0]['id'],
                                                 genre=genres[0],
                                                 release_year=track['album']['release_date'],
                                                 user=user)

            if intent == 'Recoger artista' or intent == 'Recoger artista v2':
                results = self.agent.spotify_api.search(q='artist:' + user_input, type='artist')
                if len(results['artists']['items']) == 0:
                    response = "No he podido encontrar canciones para el artista que has puesto"

                else:
                    artist = results['artists']['items'][0]
                    for album in self.agent.spotify_api.artist_albums(artist['uri'], album_type='album', limit=1)['items']:
                        for track in self.agent.spotify_api.album_tracks(album['uri'], limit=10)['items']:
                            response += '\n -' + track[
                                'name'] + ' del álbum ' + \
                                        album['name'] + ' y artista ' + \
                                        artist['name']
                            artist_full = self.agent.spotify_api.artist(artist['uri'])
                            print(artist_full['genres'])
                            available_genres = self.agent.spotify_api.recommendation_genre_seeds()['genres']
                            genres = [genre.replace(" ", "-") for genre in artist_full['genres'] if
                                      genre.replace(" ", "-") in available_genres]

                            if len(genres) > 0:
                                await self.save_song_tag(song=track['id'], artist=artist['id'],
                                                         genre=genres[0],
                                                         release_year=album['release_date'],
                                                         user=user)

            if intent == 'Recoger album' or intent == 'Recoger album v2':
                results = self.agent.spotify_api.search(q='album:' + user_input, type='album', limit=1)
                if len(results['albums']['items']) == 0:
                    response = "No he podido encontrar canciones para el álbum que has puesto"
                else:
                    for album in results['albums']['items']:
                        for track in self.agent.spotify_api.album_tracks(album['uri'], limit=10)['items']:
                            response += '\n -' + track[
                                'name'] + ' del álbum ' + \
                                        album[
                                            'name'] + ' y artista ' + \
                                        album['artists'][0]['name']
                            artist_full = self.agent.spotify_api.artist(album['artists'][0]['uri'])
                            available_genres = self.agent.spotify_api.recommendation_genre_seeds()['genres']
                            genres = [genre.replace(" ", "-") for genre in artist_full['genres'] if
                                      genre.replace(" ", "-") in available_genres]

                            if len(genres) > 0:
                                await self.save_song_tag(song=track['id'], artist=album['artists'][0]['id'],
                                                         genre=genres[0],
                                                         release_year=album['release_date'],
                                                         user=user)

            msg = Message(to="chatbot-musify@404.city")
            msg.set_metadata("performative", "inform")
            msg.body = response
            await self.send(msg)

            receive_personal_recommendation_request = PersonalRecommenderAgent.ReceivePersonalRecommendationRequest()
            template = Template()
            template.set_metadata("performative", "query")
            self.agent.add_behaviour(receive_personal_recommendation_request, template)

    class ReceivePersonalRecommendationRequest(OneShotBehaviour):
        async def run(self):
            user = await self.receive(timeout=90)
            print("Recibido el usuario " + user.body)
            b = PersonalRecommenderAgent.SendPopularRecommendationRequest()
            template = Template()
            template.set_metadata("performative", "inform")
            b.user = user.body
            self.agent.add_behaviour(b, template)
            await b.join()
            popular_recommendation = json.loads(b.exit_code)
            msg = Message(to="chatbot-musify@404.city")
            msg.set_metadata("performative", "inform")
            genres = await do_recommendation_train('genre', user.body, 'personal')
            artists = await do_recommendation_train('artist', user.body, 'personal')
            await self.put_popular_recommendation_to_personal(artists, genres, popular_recommendation['artists'])
            print(genres)
            print(artists)
            msg.body = json.dumps({"genres": genres, "artists": artists})
            await self.send(msg)

        @database_sync_to_async
        def put_popular_recommendation_to_personal(self, personal_artists, personal_genres, popular_artists):
            for artist in popular_artists:
                genres = [g['genre'] for g in list(SongTag.objects.filter(artist=artist).values("genre"))]
                genres_in_common = [g for g in genres if g in personal_genres]
                if len(genres_in_common) > 0:
                    personal_artists[random.randrange(len(personal_artists))] = artist

    class SendPopularRecommendationRequest(OneShotBehaviour):
        async def run(self):
            msg = Message(to="popular-recommender-musify@404.city")
            msg.set_metadata("performative", "query")
            msg.body = self.user
            await self.send(msg)
            popular_recommendation = await self.receive(timeout=90)
            self.exit_code = popular_recommendation.body

    async def setup(self):
        print("Personal recommender agent started")
        SPOTIPY_CLIENT_ID = '63cd1c05a2de40b19d4316d23e5271bf'
        SPOTIPY_CLIENT_SECRET = 'e0a096314a2946e4ab0c5a73f9fdd4cd'
        self.spotify_api = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                                client_secret=SPOTIPY_CLIENT_SECRET))
        b = self.ReceiveSaveSongTagsRequest()
        template = Template()
        template.set_metadata("performative", "query")
        self.add_behaviour(b, template)


class PopularRecommenderAgent(agent.Agent):
    class ReceivePopularRecommendationRequest(OneShotBehaviour):
        async def run(self):
            user = await self.receive(timeout=90)
            msg = Message(to="personal-recommender-musify@404.city")
            msg.set_metadata("performative", "inform")
            genres = await do_recommendation_train('genre', user.body, 'popular')
            artists = await do_recommendation_train('artist', user.body, 'popular')
            msg.body = json.dumps({"genres": genres, "artists": artists})
            await self.send(msg)

    async def setup(self):
        print("Popular recommender agent started")
        b = self.ReceivePopularRecommendationRequest()
        template = Template()
        template.set_metadata("performative", "query")
        self.add_behaviour(b, template)


@database_sync_to_async
def do_recommendation_train(tag, actual_user, recommender_type):
    dataset = []
    tags = []
    users = []
    print(actual_user)
    if recommender_type == 'popular':
        users = [user['user_name'] for user in list(User.objects.exclude(user_name=actual_user).values("user_name"))]
        tags = [t[tag] for t in
                list(SongTag.objects.exclude(user=User.objects.get(user_name=actual_user)).values(tag).distinct())]

    else:
        users.append(actual_user)
        tags = [t[tag] for t in
                list(SongTag.objects.filter(user=User.objects.get(user_name=actual_user)).values(tag).distinct())]

    print(users)
    print(tags)
    for user in users:
        d = list(SongTag.objects.filter(user=User.objects.get(user_name=user)) \
                 .values(tag).annotate(count=Count(tag)))

        for data in d:
            data['user'] = user

        tag_not_in_user = [t for t in tags if t not in [data[tag] for data in d]]
        for t in tag_not_in_user:
            d.append({'user': user, tag: t, 'count': 0})

        dataset.extend(d)

    if len(users) == 0:
        return []

    dataframe = pandas.DataFrame(dataset)

    r = dataframe['count'].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(r.reshape(-1, 1))
    dataframe_normalized = pandas.DataFrame(x_scaled)
    dataframe['count'] = dataframe_normalized

    matrix = dataframe.pivot(index='user', columns=tag, values='count')

    users = matrix.index.tolist()
    tags = matrix.columns.tolist()

    num_input = dataframe[tag].nunique()
    num_hidden_1 = 10
    num_hidden_2 = 5

    X = tf.placeholder(tf.float64, [None, num_input])

    weights = {
        'encoder_h1': tf.Variable(tf.random.normal([num_input, num_hidden_1], dtype=tf.float64)),
        'encoder_h2': tf.Variable(tf.random.normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
        'decoder_h1': tf.Variable(tf.random.normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
        'decoder_h2': tf.Variable(tf.random.normal([num_hidden_1, num_input], dtype=tf.float64)),
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random.normal([num_hidden_1], dtype=tf.float64)),
        'encoder_b2': tf.Variable(tf.random.normal([num_hidden_2], dtype=tf.float64)),
        'decoder_b1': tf.Variable(tf.random.normal([num_hidden_1], dtype=tf.float64)),
        'decoder_b2': tf.Variable(tf.random.normal([num_input], dtype=tf.float64)),
    }

    print(matrix)

    # Construct model

    encoder_op = encoder(X, weights, biases)
    decoder_op = decoder(encoder_op, weights, biases)

    # Prediction

    y_pred = decoder_op

    # Targets are the input data.

    y_true = X

    # Define loss and optimizer, minimize the squared error

    loss = tf.losses.mean_squared_error(y_true, y_pred)
    optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)

    predictions = pandas.DataFrame()

    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    with tf.Session() as session:
        epochs = 100
        batch_size = len(users)

        session.run(init)
        session.run(local_init)

        num_batches = int(matrix.shape[0] / batch_size)
        matrix = numpy.array_split(matrix, num_batches)

        for i in range(epochs):

            avg_cost = 0

            for batch in matrix:
                _, l = session.run([optimizer, loss], feed_dict={X: batch})
                avg_cost += l

            avg_cost /= num_batches

            print("Epoch: {} Loss: {}".format(i + 1, avg_cost))

        print("Predictions...")

        matrix = numpy.concatenate(matrix, axis=0)

        preds = session.run(decoder_op, feed_dict={X: matrix})

        predictions = predictions.append(pandas.DataFrame(preds))

        predictions = predictions.stack().reset_index(name='count')
        predictions.columns = ['user', tag, 'count']
        predictions['user'] = predictions['user'].map(lambda value: users[value])
        predictions[tag] = predictions[tag].map(lambda value: tags[value])

    print("Filtering out items in training set")

    recs = predictions
    recs = recs.sort_values(['user', 'count'], ascending=[True, False])
    if tag == 'artist':
        recs = recs.head(3)
    else:
        recs = recs.head(2)
    return recs[tag].tolist()


def encoder(x, weights, biases):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x, weights, biases):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2
