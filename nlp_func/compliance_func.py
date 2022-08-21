# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:27:51 2021

@author: JAMES.LI2
"""

import re  
import pandas as pd
from collections import Counter
import os

# Load stop words which have to be removed
compliance_stop_word_file = os.getenv("USERPROFILE") + "\\Desktop\\GitHub\\Claims-Code-Repo\\nlp_func\\dictionaries\\compliance_stop_words.csv"
compliance_stop_words = pd.read_csv(compliance_stop_word_file).Word.tolist()
count_stop_word = Counter(compliance_stop_words)

# Load compliance multigrams which will be replaced
compliance_dict_grams = {'burkina faso': 'burkinafaso',
      'costa rica': 'costarica',
     'costa rican': 'costarican',
      'costa ricans': 'costaricans',
      'east timor': 'easttimor',
      'east timorian': 'easttimorian',
      'east timorians': 'easttimorians',
      'el salvador': 'elsalvador',
      'el salvadorian': 'elsalvadorian',
      'el salvadorians': 'elsalvadorians',
      'ivory coast': 'ivorycoast',
      'puerto rican': 'puertorican',
      'puerto ricans': 'puertoricans',
      'united states': 'unitedstates',
      'las vegas': 'lasvegas',
      'los angeles': 'losangeles',
      'new york city': 'newyorkcity',
      'san antonio': 'sanantonio',
      'san diego': 'sandiego',
      'san francisco': 'sanfrancisco',
      'santa barbara': 'santabarbara',
      'washington dc': 'washingtondc',
      'land rover': 'landrover'}
multigram_re = re.compile('(%s)' % '|'.join(r'\b%s' % re.escape(mg) for mg in compliance_dict_grams.keys()))

# Replaces a multi-gram with a single word. e.g land rover -> landrover
def repl_multigram(string):
    def repl(match):
        return compliance_dict_grams[match.group(0)]
    return multigram_re.sub(repl, string)

# Takes in a string paragraph and removes all special charactersand numbers. 
# Removes capitalization and then removes all words in the stop word list
def clean_text_string_compliance(paragraph, reg = '[^A-Za-z]+'):
    word_list = []
    #reg = 
    #Replace new lines and tabs with ' ' which can cause data errors when saved as tab-delimited file.
    sent = ' '.join(paragraph.split())
    sent = sent.replace("'", '')
    sent = re.sub(reg, ' ', sent)
    sent = re.sub(r'[\s]+', ' ', sent)
    sent = sent.lower()
    #Replace compound words with token here before stemming
    sent = repl_multigram(sent)
    word_list.extend([word for word in sent.split() if not word in count_stop_word])
    return ' '.join(word_list)

# Cleans a column of a text dataframe by 
def clean_text_series_compliance(df_column):
    
    #Make a copy so the original column is not overwritten.
    text_series = df_column.copy()
    #Fill nulls with a dummy value to avoid null related errors later on
    text_series = text_series.fillna('NullString')
        
    #tqdm.pandas()
    # Use the commented if we want a progress bar
    return text_series.apply(clean_text_string_compliance)
    #return text_series.progress_apply(clean_text_string_compliance)

    