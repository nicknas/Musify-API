import requests
# Create your tests here.

def test_maria():
    register = requests.post('http://localhost:8000/register', json={'user_name': 'maria', 'password': 'maria'})
    assert register.status_code == 200
    search_metallica_album = requests.post('http://localhost:8000/maria/request_songs', json={'user_input': 'busco por el disco metallica'})
    assert search_metallica_album.status_code == 200
    print(search_metallica_album.json())
    metallica_recommendation = requests.get('http://localhost:8000/maria/request_recommendations')
    assert metallica_recommendation.status_code == 200
    print(metallica_recommendation.json())
    search_eminem_artist = requests.post('http://localhost:8000/maria/request_songs', json={'user_input': 'dime canciones del cantante eminem'})
    assert search_eminem_artist.status_code == 200
    print(search_eminem_artist.json())
    eminem_recommendation = requests.get('http://localhost:8000/maria/request_recommendations')
    assert eminem_recommendation.status_code == 200
    print(eminem_recommendation.json())

def test_eva():
    register = requests.post('http://localhost:8000/register', json={'user_name': 'eva', 'password': 'eva'})
    assert register.status_code == 200
    search_50cent_artist = requests.post('http://localhost:8000/eva/request_songs', json={'user_input': 'busco temas del artista 50 cent'})
    assert search_50cent_artist.status_code == 200
    print(search_50cent_artist.json())
    cent_recommendation = requests.get('http://localhost:8000/eva/request_recommendations')
    assert cent_recommendation.status_code == 200
    print(cent_recommendation.json())

test_maria()
test_eva()