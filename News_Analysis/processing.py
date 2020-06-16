# import os
# from django.core.files.storage import default_storage
import os
from .details import Details
from .ocr import ocr
from .spark import sparkfun
import cv2
import numpy
from .models import ImageAnalysis, TextFilesAnalysis
import codecs
from PIL import Image
import os.path

# import asyncio


def image_processing(image, spark, model, row):
    img = cv2.imdecode(numpy.fromstring(image.file.getvalue(), numpy.uint8), cv2.IMREAD_COLOR)
    name = image.name
    text = ocr(img)
    list_ = [{"name": name, "text": text, "label": 2}]
    result = sparkfun(list_, spark, model, row)
    # name_ = result['name']
    text_ = result['text']
    prediction_ = result['prediction']
    word_cloud = result['word_cloud']
    word_frequency_graph = result['word_frequency_graph']
    try:
        instance = ImageAnalysis(name=name, text=text_[0],
                                 prediction=prediction_, image=image)
        instance.word_cloud.save(name, word_cloud)
        instance.word_frequency_graph.save(name, word_frequency_graph)
        instance.save()
    except:
        print('Duplicate')
    # word_cloud = Image.open(instance.word_cloud.path)
    # word_cloud.thumbnail(100, 100)
    # original_image = Image.open(instance.image.path)
    # original_image.thumbnail(100, 100)
    # obj = Details(name, text_[0], prediction_)
    # obj.text = text_[0]
    # obj.name = name
    # obj.prediction = prediction_
    # obj.image =original_image
    # obj.word_cloud = word_cloud


def text_files_processing(file, spark, model, row):
    name = file.name
    try:
        text = file.read()
    except:
        print("File is corrupted")
        return
    list_ = [{"name": name, "text": codecs.decode(text, errors='ignore'), "label": 2}]
    result = sparkfun(list_, spark, model, row)
    # name_ = result['name']
    text_ = result['text']
    prediction_ = result['prediction']
    word_cloud = result['word_cloud']
    word_frequency_graph = result['word_frequency_graph']
    try:
        instance = TextFilesAnalysis(name=name, text=text_[0],
                                     prediction=prediction_, text_file=file)
        pre, ext = os.path.splitext(name)
        instance.word_cloud.save(pre + ".png", word_cloud)
        instance.word_frequency_graph.save(pre + ".png", word_frequency_graph)
        instance.save()
    except:
        print("Duplicate Data")
