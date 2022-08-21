# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:07:23 2022

@author: RUI.WANG2
"""
import pandas as pd
import pyodbc
import pickle

df = pd.read_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\indvid_notes_sql3.pkl")
print(len(df))
Data = df.groupby(['CLM_NUM'])['NOTES'].apply(list).reset_index()
Data.to_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\combined_notes_sql3.pkl")