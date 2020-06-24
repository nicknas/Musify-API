import numpy
import pandas
import random
from django.db.models import Count
from sklearn import preprocessing
from spade import agent
from spade.behaviour import *
import json
import tensorflow as tf

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
            recommendations = await self.receive(timeout=15)
            if recommendations is not None:
                self.exit_code = recommendations.body
            else:
                self.exit_code = "No se han podido generar recomendaciones"

    async def setup(self):
        print("Chatbot agent started")
        b = self.SendPersonalRecommenderRequest()
        b.user = self.user
        self.request_recommendation = b
        template = Template()
        template.set_metadata("performative", "inform")
        self.add_behaviour(b, template)


class PersonalRecommenderAgent(agent.Agent):
    class ReceivePersonalRecommendationRequest(OneShotBehaviour):
        async def run(self):
            user = await self.receive(timeout=15)
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
            popular_recommendation = await self.receive(timeout=15)
            self.exit_code = popular_recommendation.body

    async def setup(self):
        print("Personal recommender agent started")
        b = self.ReceivePersonalRecommendationRequest()
        template = Template()
        template.set_metadata("performative", "query")
        self.add_behaviour(b, template)


class PopularRecommenderAgent(agent.Agent):
    class ReceivePopularRecommendationRequest(OneShotBehaviour):
        async def run(self):
            user = await self.receive(timeout=15)
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
