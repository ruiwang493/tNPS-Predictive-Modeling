# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 09:10:17 2022

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
Data = pd.read_sql_query(
    '''
    With --8176
tnps_CTE as
(	
	SELECT * FROM ( 
		SELECT DISTINCT CLM.NK_CLM_ID, CLM.CLM_RPRTD_DTTM, CLM.CLM_LOSS_DTTM,CLM.LOSS_CSE_TYP_CD ,CLM.CLM_DESC_TXT, CLM.CLM_STS_CD,
		SRVY.ETL_SELECTION_DATE,
		CONCAT(RSPS.CLAIM_NUMBER ,'-1') AS CLM_NUM
		,CASE 
			WHEN (SRVY.SURVEY_CATEGORY = 'V6' AND RSPS.ANSWER_20 >= 9) THEN 'Promoter'
			WHEN (SRVY.SURVEY_CATEGORY = 'V6' AND RSPS.ANSWER_20 IN ('8','7')) THEN 'Passive'
			WHEN (SRVY.SURVEY_CATEGORY = 'V6' AND RSPS.ANSWER_20 < 7) THEN 'Detractor'
			WHEN (UPPER(SRVY.SURVEY_CATEGORY) = 'V6_GLASS' AND RSPS.ANSWER_2 >= 9) THEN 'Promoter'
			WHEN (UPPER(SRVY.SURVEY_CATEGORY)= 'V6_GLASS' AND RSPS.ANSWER_2 IN ('8','7')) THEN 'Passive'
			WHEN (UPPER(SRVY.SURVEY_CATEGORY) = 'V6_GLASS' AND RSPS.ANSWER_2 < 7) THEN 'Detractor'
		 END AS LTR_DESIGNATION
		,CAST(CASE 
				WHEN (SRVY.SURVEY_CATEGORY = 'V6' AND RSPS.ANSWER_20 >= 9) THEN '100'
				WHEN (SRVY.SURVEY_CATEGORY = 'V6' AND RSPS.ANSWER_20 IN ('8','7')) THEN '0'
				WHEN (SRVY.SURVEY_CATEGORY = 'V6' AND RSPS.ANSWER_20 < 7) THEN '-100'
				WHEN (UPPER(SRVY.SURVEY_CATEGORY) = 'V6_GLASS' AND RSPS.ANSWER_2 >= 9) THEN '100'
				WHEN (UPPER(SRVY.SURVEY_CATEGORY) = 'V6_GLASS' AND RSPS.ANSWER_2 IN ('8','7')) THEN '0'
				WHEN (UPPER(SRVY.SURVEY_CATEGORY) = 'V6_GLASS' AND RSPS.ANSWER_2 < 7) THEN '-100'
			 END AS INT) AS tNPS
		, RSPS.RESPONSE_DATE
		, ROW_NUMBER() OVER (PARTITION BY RSPS.CLAIM_NUMBER ORDER BY RSPS.CDH_UPDATE_DT DESC) AS RN
	FROM CDH_Store..CDH_INET_DP_RESPONSES_PERM RSPS with (nolock)
		INNER JOIN CDH_Store..CDH_INET_SURVEY_CONTACTS SRVY with (nolock)ON 1=1 
			AND SRVY.DOCUMENT_ID = RSPS.DOCUMENT_ID
		INNER JOIN ODSBase_Store..CLM CLM with (nolock) ON 1=1
    		AND CONCAT(RSPS.CLAIM_NUMBER ,'-1') = CLM.CLM_NUM
    		AND CLM.CLM_LOB_TYP_CD IN  ('PersonalAutoLine' , 'BusinessAutoLine')
			and RSPS.RESPONSE_DATE >= '2021-07-01' and RSPS.RESPONSE_DATE < '2022-07-01'
    		--AND CLM.clm_num = '7000309017-1'
	WHERE (UPPER(SRVY.SURVEY_CATEGORY) = 'V6_GLASS' AND RSPS.ANSWER_2 IS NOT NULL) 
			OR (SRVY.SURVEY_CATEGORY = 'V6' AND RSPS.ANSWER_20 IS NOT NULL)
	) t1
	WHERE RN=1
)
, 
payment_cte as
(
	select * from
	(
		SELECT tnps.LTR_DESIGNATION, tnps.RESPONSE_DATE, tnps.tNPS,tnps.ETL_SELECTION_DATE,
			CDH_GW_CLM.CLM_NUM
			, CDH_GW_CLM.NK_CLM_ID
			,CDH_GW_CLM.CLM_RPRTD_DTTM 
			,CDH_GW_TRN.EXPSR_COST_CATG_TYP_CD  
			,CDH_GW_TRN.PYMT_TYP_CD 
			,CDH_GW_TRN.CLM_EXPSR_TRN_SBTYP_CD 
			,CDH_GW_TRN.SYSTM_GNRTD_IND 
			,CDH_GW_TRN_LN_ITM.TRN_AMT 
			,CDH_GW_TRN.ODS_EFF_STRT_DTTM as TRN_ODS_EFF_STRT_DTTM
			,CDH_GW_TRN.CLM_EXPSR_TRN_STS_TYP_CD 
			,CDH_GW_TRN_LN_ITM.ODS_EFF_STRT_DTTM as LN_ITM_ODS_EFF_STRT_DTTM
			,CDH_GW_TRN.NK_CHK_ID 
			, ROW_NUMBER() OVER (PARTITION BY CDH_GW_CLM.CLM_NUM 
					ORDER BY CDH_GW_TRN_LN_ITM.ODS_EFF_STRT_DTTM, CDH_GW_TRN.ODS_EFF_STRT_DTTM) AS RN
		FROM tnps_CTE tnps
			inner join ODSBase_Store..CLM CDH_GW_CLM with (nolock) on CDH_GW_CLM.CLM_NUM = tnps.CLM_NUM
				LEFT JOIN ODSBase_Store..CLM_EXPSR CDH_GW_CLM_EXPSR with (nolock) ON CDH_GW_CLM.NK_CLM_ID = CDH_GW_CLM_EXPSR.NK_CLM_ID
				LEFT JOIN ODSBase_Store..CLM_EXPSR_TRN CDH_GW_TRN with (nolock) ON CDH_GW_CLM_EXPSR.NK_CLM_EXPSR_ID = CDH_GW_TRN.NK_CLM_EXPSR_ID
				LEFT JOIN ODSBase_Store..CLM_EXPSR_TRN_LN_ITM CDH_GW_TRN_LN_ITM with (nolock) ON CDH_GW_TRN.NK_CLM_EXPSR_TRN_ID = CDH_GW_TRN_LN_ITM.NK_CLM_EXPSR_TRN_ID            
	   
		WHERE CDH_GW_TRN.clm_expsr_trn_sts_typ_cd ='submitted' 
			AND CDH_GW_TRN.CLM_EXPSR_TRN_SBTYP_CD = 'Payment'
			AND CDH_GW_TRN_LN_ITM.TRN_AMT  > 0 
	) as pym
	where rn = 1
)
,--145016
notes_CTE as (
SELECT 
	p.CLM_NUM
	,p.NK_CLM_ID 
	,p.CLM_RPRTD_DTTM 
	,p.ETL_SELECTION_DATE
	,p.RESPONSE_DATE  
	,NOTE.NK_NOTE_ID 
	,NOTE.ODS_EFF_STRT_DTTM as NOTE_TRN_CRTD_DTTM 
	,NOTE.BODY_TXT AS NOTES 
	,NOTE.NK_CRT_USR_ID 
	,p.LN_ITM_ODS_EFF_STRT_DTTM
		FROM payment_cte p
		LEFT JOIN ODSBase_Store..NOTE NOTE with (nolock)
			ON p.nk_clm_id = NOTE.nk_clm_id 
)
select * from notes_CTE 
order by CLM_NUM asc
    ''', conn)
Data.to_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\indvid_notes_sql3.pkl")
'''
curr = Data.loc[:,"CLM_NUM"][0]
Data2 = Data.copy()
for i in range(1,len(Data)):
    currNote = Data.loc[:,"NOTES"][i] + "| "
    print(curr, currNote)
    if Data.loc[:,"CLM_NUM"][i] == curr:
        Data2.at[Data2.loc[:,"CLM_NUM"].index(curr),"NOTES"] += currNote
    else:
        curr = Data.loc[:,"CLM_NUM"][i]
print(Data2.loc[:,"NOTES"][0])

Data2.drop_duplicates(subset = "CLM_NUM")
'''
#Data2.to_pickle("C:\\Users\\rui.wang2\\Box\\Claims Adv Analytics\\Data Scientists\\Rui Wang\\proj\\output\\notes_sql.pkl")