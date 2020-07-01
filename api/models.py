from django.db import models


class User(models.Model):
    """
    Clase que define la tabla Usuario de la base de datos, junto con su DAO
    """
    user_name = models.CharField(primary_key=True, max_length=50)
    password = models.CharField(max_length=50, blank=False)


class SongTag(models.Model):
    """
    Clase que define la tabla Etiquetas de la base de datos, junto con su DAO
    """
    artist = models.CharField(max_length=50, blank=False)
    genre = models.CharField(max_length=50, blank=False)
    release_year = models.CharField(max_length=50, blank=False)
    song = models.CharField(max_length=50, blank=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
