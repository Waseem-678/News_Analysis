import re
from pytesseract import image_to_string, image_to_data, Output
import cv2
import numpy as np
import pkg_resources
from symspellpy.symspellpy import SymSpell


def conf(img):
    d = image_to_data(img, output_type=Output.DICT)
    test_list = d["conf"]
    confidence = []
    for i in range(0, len(test_list)):
        temp = int(test_list[i])
        if temp == -1:
            continue
        else:
            confidence.append(temp)

    return np.mean(confidence)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def postprocessing(text):
    max_edit_distance_dictionary = 2
    prefix_length = 7
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    # load dictionary
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    if not sym_spell.load_dictionary(dictionary_path, term_index=0,
                                     count_index=1):
        print("Dictionary file not found")
        return
    if not sym_spell.load_bigram_dictionary(dictionary_path, term_index=0,
                                            count_index=2):
        print("Bigram dictionary file not found")
        return

    result = sym_spell.word_segmentation(text.lower())
    return result.corrected_string


# def remove_pun(text):
#     punctuations = '''!()-[]{};:\'\"\\,`"'<>./?@#$%^&*_~'''
#     no_punct = ""
#     for char in text:
#         if char not in punctuations:
#             no_punct = no_punct + char
#
#     return no_punct


def ocr(img):
    confidence = conf(img)
    if confidence >= 88:
        temp = image_to_string(img, lang='eng')
        temp1 = postprocessing(temp)
        # res = re.sub('[^A-Za-z0-9]+', ' ', temp1).strip()
        return temp1
    img = cv2.fastNlMeansDenoisingColored(img, img, 3, 3, 7, 21)
    gray = get_grayscale(img)
    dst = gray
    cv2.addWeighted(gray, 0.58, gray, 0.5, 0, dst)
    gama = adjust_gamma(dst, 2.2)
    cv2.fastNlMeansDenoising(gama, gama, 10, 7, 21)
    gama = cv2.GaussianBlur(gama, (1, 1), 0, 0)
    cv2.adaptiveThreshold(gama, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 20)
    kernel = np.ones((3, 3), np.uint8)
    gama = cv2.erode(gama, kernel, iterations=1)
    gama = cv2.dilate(gama, kernel, iterations=1)
    text = image_to_string(gama, lang='eng')
    temp = postprocessing(text)
    # res = re.sub('[^A-Za-z0-9]+', ' ', temp).strip()
    return temp

