# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:49:13 2022

@author: RUI.WANG2
"""
import pandas as pd

df = pd.read_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\dataset.pkl")

#df['LTR_DESIGNATION'].value_counts()
'''
Promoter: 61064
Passive: 7110
Detractor: 5373
Proportion of Promoter: 0.83
Proportion of Detractor: 0.07
tNPS Score: 75.7


Proportion of claims without notes: 0.475 (34904)
Proportion of claims with notes:    0.525 (￼38643)
'''

#bar graph of values/percentages
ax = df['LTR_DESIGNATION'].value_counts().plot(kind='bar', figsize=(6,6))
fig = ax.get_figure()
ax.set_title("Breakdown of tNPS designation (Promoter/Passive/Detractor")
ax.set_xlabel('Designation')
ax.set_ylabel('Value Counts')



