# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 19:37:32 2021

@author: William
"""

import pandas as pd
import os
import numpy as np
import cv2
import pytesseract
import enchant
from spellchecker import SpellChecker
import gensim
from gensim.models import KeyedVectors



PATH = r"part2\26953_Bebe"

#Read the dataframe
metadata = pd.read_csv("data.csv")

angle_to_int = {"avant": [1, 11, 31], 'gauche': [2, 14, 15, 32], "droite": [3, 23, 24, 25, 35], "dessus": [4, 6, 17, 18, 19, 26, 27, 28, 33, 36
], "arrière": [5, 20, 21, 22, 34], "3/4": [7, 12, 13, 16], "autre": [8, 9, 10, 11, 29, 30] }

#Build dictionary to get the angle from the int representation
int_to_angle = {}
for key, value in angle_to_int.items():
    for n in value:
        int_to_angle[n] = key


def get_angle(file):
    #Get angle from the filename
    
    s = file.split('_')
    return int_to_angle[int(s[2])]

def get_name(file):
    #Get name from the filename
    
    s = file.split('_')
    return int(s[1])



def get_level(barcode):
    
    target_row =  metadata.loc[metadata['barcode']==barcode]
    return target_row['level3'].values[0], target_row['nodeid3'].values[0]




def show(image):
    #image = cv2.resize(image, (960, 540))
    #cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
    #imS = cv2.resize(image, (960, 540))  
    cv2.imshow('image', image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ocr(file):
    img = cv2.imread(os.path.join(PATH, file))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((2, 1), np.uint8)
    img = cv2.erode(gray, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    out_below = pytesseract.image_to_string(img)
    
    return out_below

import re
def text_preprocess(s):
    '''Nettoie le texte'''
    
    s = s.replace("\n", " ")
    s = s.replace("\\\\", " ")
    s = re.sub('[^A-Za-z0-9]+', ' ', s)
    return s

def prune_french(s):
    '''Corrige les erreurs de l'OCR et conserve uniquement les mots en français'''
    spell = SpellChecker(language='fr')
    l = s.split(" ")
    new_l = list(map(lambda x: spell.correction(x), l))
    for i in range(len(new_l)):
        if new_l[i] not in spell:
            new_l[i] = ""
    return " ".join(new_l)

stopwords = pd.read_json('stop_words_french.json')

def remove_stopwords(s):
    '''Enlève les stopwords'''
    return " ".join([word for word in s.split() if word not in stopwords.values])

model = KeyedVectors.load_word2vec_format("frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin", binary=True, unicode_errors="ignore")


def model_dist(words, categorie):
    
    
    l = list(map(lambda y : model.distance(categorie, y),  words))
    if not l == []:
        return np.argmax(l)
    else:
        return 0
    


if __name__ == '__main__':
    data = pd.DataFrame({'name': os.listdir(PATH)})
    data['angle'] = data['name'].apply(get_angle)
    data['barcode'] = data['name'].apply(get_name)
    data['level3'] = data['barcode'].apply(lambda x: get_level(x)[0])
    data['nodeid3'] = data['barcode'].apply(lambda x: get_level(x)[1])
    
    s = (ocr("image_3041090000263_2_18fd3970.jpeg"))    
    
    s_preprocess = text_preprocess(s)
    
    s_prune = prune_french(s_preprocess)
    
    s_stopwords = remove_stopwords(s_prune)
    
    print(s)
    print(s_stopwords)
   
    