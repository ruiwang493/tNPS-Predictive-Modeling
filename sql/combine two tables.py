# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 06:59:47 2022

@author: RUI.WANG2
"""
import pandas as pd
import functions as func
from nltk.stem import WordNetLemmatizer

# read in the pickle files for the combined notes as well as the payment information
df = pd.read_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\combined_notes_sql3.pkl")
df2 = pd.read_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\payments_sql.pkl")

# combine the two tables and remove the duplicate CLM_NUM row
df3 = pd.concat([df, df2], axis=1, join='inner')
df3 = df3.loc[:,~df3.columns.duplicated()].copy()

# cast the 'NOTES' column as a string (previously was a list due to how the notes were combined) and clean the notes
df3 = df3.astype({'NOTES':'string'})
df3['NOTES'] = list(map(func.clean_text, df3.NOTES))

# lemmatize the NOTES column
def lemmatized_words(text):
    lemm = WordNetLemmatizer()
    df3['lemmatized_text'] = list(map(lambda word:list(map(lemm.lemmatize, word)), df3.NOTES))
lemmatized_words(df3.NOTES)

'''
df3['LTR_DESIGNATION'] = df3['LTR_DESIGNATION'].replace(['Passive', 'Detractor'], 0)
df3['LTR_DESIGNATION'] = df3['LTR_DESIGNATION'].replace(['Promoter'], 1)
df3['PYMT_TYP_CD'] = df3['PYMT_TYP_CD'].replace(['final'], 1)
df3['PYMT_TYP_CD'] = df3['PYMT_TYP_CD'].replace(['partial', 'supplement'], 0)
df3['NOTES'] = func.clean_text_compliance(df3['NOTES'] , include_spec_char = "")  
'''

#df3.to_excel("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\dataset.xlsx")
#df3.to_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\dataset.pkl")
