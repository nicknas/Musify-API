from django.db import models

class User(models.Model):
    user_name = models.CharField(primary_key = True, max_length = 50)
    password = models.CharField(max_length = 50, blank = False)

class SongTag(models.Model):
    artist = models.CharField(max_length = 50, blank = False)
    genre = models.CharField(max_length = 50, blank = False)
    release_year = models.PositiveSmallIntegerField(default = 0)
    album = models.CharField(max_length = 50, blank = False)
    user = models.ForeignKey(User, on_delete = models.CASCADE)
