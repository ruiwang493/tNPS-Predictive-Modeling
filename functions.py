# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:57:56 2022

@author: RUI.WANG2
"""

import re  
from symspellpy import SymSpell, Verbosity
import csv
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer
from collections import Counter
import glob_var
glob_var.init()

def clean_text(text, remove_stopwords = True):
    '''Text Preprocessing '''
    
    # Convert words to lower case
    text = text.lower()
    
    # Expand contractions
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in glob_var.contractions:
                new_text.append(glob_var.contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    reg = '[^A-Za-z]+'
    text = re.sub(reg, " ", text)
    text = re.sub(r"\s+"," ",text) 
    
    
    #removing conventional stopwords and compliance stopwords
    stops = set(stopwords.words('english'))
    if remove_stopwords:
        text = text.split()
        text = [w for w in text if not w in stops]
        text = [w for w in text if not w in glob_var.compliance_stop_words]
        text = " ".join(text)
        
    # Tokenize each word
    text = nltk.WordPunctTokenizer().tokenize(text)
    return text
