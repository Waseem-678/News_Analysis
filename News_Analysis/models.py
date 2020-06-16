from django.db import models


class ImageAnalysis(models.Model):
    name = models.CharField(max_length=100)
    text = models.TextField(blank=True, null=True, unique=True)
    prediction = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/', null=True, blank=True, unique=True)
    word_cloud = models.ImageField(upload_to='images/word_cloud/', null=True, blank=True)
    word_frequency_graph = models.ImageField(upload_to='images/word_frequency_graph/', null=True, blank=True)

    def __str__(self):
        return self.name


class TextFilesAnalysis(models.Model):
    name = models.CharField(max_length=100)
    text = models.TextField(blank=True, null=True, unique=True)
    prediction = models.CharField(max_length=100)
    text_file = models.FileField(upload_to='text_files/', null=True, blank=True, unique=True)
    word_cloud = models.ImageField(upload_to='text_files/word_cloud/', null=True, blank=True, unique=True)
    word_frequency_graph = models.ImageField(upload_to='text_files/word_frequency_graph/', null=True, blank=True)

    def __str__(self):
        return self.text_file.name


