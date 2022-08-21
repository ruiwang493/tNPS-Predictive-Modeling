# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 07:34:57 2022

@author: RUI.WANG2
"""

import os
import datetime as dt
import csv

def init():
    global GIT_DIR, OUTPUT_DIR, currentDate, log, claim_key, contractions, compliance_stop_words, compliance_dict_grams
    
    currentDate = dt.datetime.today()
    
    # Directories
    GIT_DIR = os.getenv("USERPROFILE") + "\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj"
    OUTPUT_DIR = os.getenv("USERPROFILE") + "\\Box\\CLaims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output"
    log = OUTPUT_DIR + "logs\\" + str(currentDate.year) + "_" + str(currentDate.month) + "_" + str(currentDate.day) + ".txt"
    
    # Keywords
    claim_key = "NK_CLM_ID"
    
    # SQL
    
    # Dictionaries
    compliance_stop_words = []
    with open(GIT_DIR + "\\dictionary\\compliance_stop_words.csv", 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        for row in reader:
            compliance_stop_words.append(row[0])
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
    contractions = {}
    with open(GIT_DIR + "\\dictionary\\word-replacement.csv", 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for row in reader:
            contractions[row[0]] = row[1]
init()