# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:50:57 2022

@author: RUI.WANG2
"""
import pandas as pd
import pyodbc
import pickle
conn = pyodbc.connect('Driver={SQL Server};'
'Server=fewvpsdb1111;'
'Database=master;'
'Trusted_Connection=yes;')

cursor = conn.cursor()

def getNote(claim_Number):
    x = "'" + claim_Number + "'"
    Data = pd.read_sql_query('''SELECT
    CLAIM.CLM_NUM
    ,NOTE_CATG_TYP_CD

    ,CASE
      WHEN note.nk_mttr_id IS NOT NULL THEN 'Matter'
      WHEN note.nk_srvc_request_id IS NOT NULL THEN 'Service'
      WHEN note.nk_prty_clm_id IS NOT NULL THEN 'Party'
      WHEN note.nk_clm_expsr_id IS NOT NULL THEN 'Exposure'
      ELSE 'Claim'
    END AS NOTE_TYPE

    ,note.NOTE_TOPIC_TYP_CD
    ,note.BODY_TXT as NOTES
    ,note.ORGNL_ATHRD_DTTM

    FROM 

    ODSBase_Store.dbo.CLM claim

    INNER JOIN ODSBase_Store.dbo.NOTE NOTE
    ON claim.nk_clm_id = NOTE.nk_clm_id

    WHERE claim.clm_num = ''' + x + '''
    ORDER BY NOTE.ORGNL_ATHRD_DTTM DESC''', conn)
    notes = Data.loc[:,"NOTES"]
    s = ''
    for i in range(len(notes)):
        s += str(notes[i]) + " "
    return s

