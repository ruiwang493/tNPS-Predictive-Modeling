**Predicting tNPS for Personal Auto Cases**
=
by Rui Wang
## Content Overview
- **dictionary**: Dictionaries I used for text cleaning for NOTES data.
- **figures**: All the relevant figures for my presentation.
- **nlp_func**: Example NLP text-processing functions that helped me/served as inspiration.
- **output**: All the output of the project, including:
	- datasets
	- training results
	- confusion matrices
- **pkg**: Package library that I wrote for my data preprocessing and text cleaning, as well as global variables, etc.
- **sql**: All the SQL queries that I wrote and used to extract my datasets.
- *init*.*py*: To initialize and call all the functions and variables in my package library
- *baseline_logreg_model.py*: The baseline model I wrote, which is a standard logistic regression classification.
- *pca*.*py*: Second model that I wrote, improvement on the baseline logistic regression model, this time using logistic regression with PCA.

## Data - (Excluded on GitHub due to sensitivity and confidential information)
All found within the output folder. The files that have .xlsx (Excel spreadsheets) of the same name are as such because of DataRobot. DataRobot does not accept .pkl files as valid datasets for its projects, so I simply copied the .pkl files below into .xlsx ones to use for DataRobot.
- **dataset**: The main dataset that I used after all the preprocessing and text cleaning and NOTES concatenation. This is the model that I directly loaded up for my models to use.
- **training**: Since I was going to be training multiple models, it was important for me to standardize and have a universal training data set, which is this one. Similarly for **testing**.
- **testing**: The testing portion of **dataset**, for more information, see **training**.
- *individ_notes_sql*: the files with 1 and 2 at the end are early, obsolete versions, as *individ_notes_sql3* is the final version. This dataset is the dataset I pulled from MS SQL, with individual NOTES entries being entirely new rows.
- *combined_notes_sql*: the files with 1 and 2 at the end are early, obsolete versions, as *combined_notes_sql3* is the final version. This dataset is the dataset I pulled from MS SQL after applying the logic to concatenate all the NOTES for each claim (using python).
- *payments_sql*: The payments data that I pulled from MS SQL after sorting by claim number.

## SQL
- **tnps_sql_server**: Query that would get all the data, with notes concatenated. Later abandoned due to how slowly it completed (taking up to some 6 hours). Replaced in functionality by **tnps_sql_server (tnp and payment dates)** and **tnps_sql_server (tnp and individual notes)**, which were combined together in python.
- **tnps_sql_server (tnp and payment dates)**: Query that got the payment half of the dataset. 
- **tnps_sql_server (tnp and individual notes)**: Query that got the notes half of the dataset. Notes are not concatenated.
- **payments_sql.py**: *tnps_sql_server (tnp and payment dates)* but run through python. Output is *payments_sql* in  **Data**
- **notes_sql.py**: *tnps_sql_server (tnp and individual notes)* but run through python. Output is the *individ_notes_sql* files in **Data**
- **combine notes for sql**: Python code that combined the notes in the *individ_notes_sql* files. Outputs are the *combined_notes_sql* files.
- **compile Note for specific CLM_NUM**: Early python function I wrote serving as the groundwork for **combine notes for sql**. Obsolete.
- **combine two tables**: Python code that combined *combined_notes_sql* and *payments_sql* to create the first form of *dataset*.

## Models
I wrote and implemented 2 models, a baseline logistic regression classification model, as well as logistic regression model with PCA. The other models that I used were from DataRobot.
- Baseline Logistic Regression Model:
	- The first model I wrote. Uses bag-of-words and tfidf transformations on the concatenated NOTES data, in conjunction with the other features like payment amount, and exposure category. I ran the classification on both just the bag-of-words transformed data as well as the tfidf transformed data.
- Logistic Regression Model with PCA:
	- My thought process in using PCA was to try and improve on the baseline model, and implementing PCA helped solve some of the issues I had with regards to multicollinearity that I was having with the baseline model. Since tfidf performed better in the baseline model, I thought to run the model with PCA on just tfidf-transformed data.
- DataRobot
	- DataRobot produced many other models. The best performing one with regards to AUC was an Extreme Gradient Boosted Decision Tree Classifier.
