# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:06:53 2021

@author: JAMES.LI2

Fuzzy word search code originated from Ying
"""
#Has the fuzzy search functions
import numpy as np   
import pandas as pd
from fuzzywuzzy import fuzz 
import Levenshtein #DONT DELETE! SIGNIFICANTLY IMPROVES FUZZY MATCHING

# Takes in a keyword list and then a list of strings to search for
def fuzzy_word_search(keyword_list, list_to_search_in,fuzz_ratio = 88):
    print('Start fuzzy searching for keywords...')
    match_dict=dict()
    match_list=list()

    for keyword in keyword_list:
        for singleword in list_to_search_in:
            if fuzz.ratio(str(singleword),keyword)>=fuzz_ratio:
                match_list.append(singleword)
        match_dict[keyword]=match_list
        match_list=list()    
        print ('Finished keyword: ' + keyword)
    return match_dict

def reverse_dictionary(dictionary):
    reverse_dictionary = dict()
    for key in dictionary.keys():
        for word in dictionary[key]:
           reverse_dictionary[word] = key
    return reverse_dictionary

# This function fuzzy match the list of keywords in keyphrases file to the words from data frame df's text fields.
# df - data frame name
# text_field - column name of the text field to be searched from
# Keyphrases - keyphrase file includes words to be searched for and their groupings
# level_key - roll up level - eg. claim exposure number
# excluded - list of words in the document whose "original spelling" should not count as a match to the keyword
def FuzzyWordSearch(df, text_field: str, Keyphrases, level_key: str, excluded: list, fuzz_ratio = 88):
    #define the key phrases to search for 
    #read in the file with key phrases to search for
    Keyphrases['Number_of_Words'] = Keyphrases['Keyphrase'].apply(lambda x : len(x.split()))
    max_number_of_words = max(Keyphrases['Number_of_Words'])
    keyword_list = ' '.join(list(Keyphrases['Keyphrase'])).split()
    displayed_keyphrase_list = list(set(Keyphrases['Displayed_Keyphrase']))
    df = df.copy()
    
    # convert text to be one word per row for next few operations
    # Assume that the text field has already been cleaned
    df['word_list'] = df[text_field].apply(lambda x : x.split())
    df['INDEX'] = df.index
    Long_Data = pd.DataFrame([(row.INDEX, word) for row in df.itertuples() for word in row.word_list], columns = ['INDEX', 'Original_Word'])
    Long_Data = Long_Data.reset_index()
    Long_Data.rename(columns = {'index' : 'Word_Order'}, inplace = True)
            
    all_unique_words = list(set(' '.join(df[text_field]).split(' ')))     #create a list of all unique words in the dataset. Fuzzy matching is slow so running it on each unique word once improves performance
    # Create a unique keyword list too
    keyword_list=list(set(keyword_list))
    keyword_to_unique_dict = fuzzy_word_search(keyword_list, all_unique_words, fuzz_ratio)     #create a dictionary where the key is a word in the keyword list and the values are a list of unique spellings that fuzzy-matched it
    unique_word_dict = reverse_dictionary(keyword_to_unique_dict)     #reverves the dictionary just created so that we have a dictionary where the keys are the unique spellings and the values are the words from the keyword list
    ############################################################################################################################################################################
    # Create a dataframe that maps all accepted unique spelling of a keyphrase to the keyphrase
    ############################################################################################################################################################################    
    Unique_Words = pd.Series(unique_word_dict).reset_index()
    Unique_Words.rename(columns = {'index' : 'Original_Word', 0 : 'Keyword'}, inplace = True)    #!#
    
    Long_Data = Long_Data.merge(Unique_Words, on = 'Original_Word', how = 'left')  #Changed inner to left join - Jan30,2020
    Long_Data.sort_values('Word_Order', inplace = True)
    
    ############################################################################################################################################################################
    # Create columns that the words shifted "up" by 1, 2, . . . n. These columns will be used to create columns of ngrams (phrases of n words)
    ############################################################################################################################################################################     
    for n in range(1, max_number_of_words):
        Long_Data['Shift' + str(n)] = Long_Data['Original_Word'].shift(-1*n) 
        Long_Data['Keyword_Shift' + str(n)] = Long_Data['Keyword'].shift(-1*n) 
    ##Shift down by 1 to get the prior word
    Long_Data['Shift' + str(-1)] = Long_Data['Original_Word'].shift(1) 
    Long_Data['Keyword_Shift' + str(-1)] = Long_Data['Keyword'].shift(1)     
    
    Long_Data.rename(columns = {'Original_Word' : '1gram', 'Keyword' : 'Keyword_1gram'}, inplace = True)
    
    ############################################################################################################################################################################
    # Create "ngrams" (combination of 2 words, 3 words, etc.) columns
    ############################################################################################################################################################################             
    for n in range(2, max_number_of_words+1):
        Long_Data[str(n) + 'gram'] = Long_Data['1gram']
        Long_Data['Keyword_' + str(n) + 'gram'] = Long_Data['Keyword_1gram']
        for subindex in range(1, n):
            Long_Data[str(n) + 'gram'] = Long_Data[str(n) + 'gram'] + ' ' + Long_Data['Shift' + str(subindex)]
            Long_Data['Keyword_' + str(n) + 'gram'] = Long_Data['Keyword_' + str(n) + 'gram'] + ' ' + Long_Data['Keyword_Shift' + str(subindex)]
    
    ############################################################################################################################################################################
    # Merge on the key phrases that we are looking for . . . the merge by column will be different depending on how many words are in the keyphrase
    ############################################################################################################################################################################     
    Long_Data['Matched_Keyphrase'] = ''
    Long_Data['Matched_Keyphrase_Original_Spelling'] = ''    
    
    for n in range(max_number_of_words):
        Subset_Keyphrases = Keyphrases.loc[Keyphrases['Number_of_Words'] == n+1] 
        Long_Data = Long_Data.merge(Subset_Keyphrases[['Keyphrase']], how = 'left', left_on = 'Keyword_' + str(n+1)+'gram', right_on = 'Keyphrase')
        Long_Data.rename(columns = {'Keyphrase' :  str(n+1) + 'gram_Matched'}, inplace = True)        
        Long_Data['Matched_Keyphrase'] = np.where((pd.isnull(Long_Data[str(n+1) + 'gram_Matched'])), Long_Data['Matched_Keyphrase'],Long_Data[str(n+1) + 'gram_Matched'])
        Long_Data['Matched_Keyphrase_Original_Spelling'] = np.where((pd.isnull(Long_Data[str(n+1) + 'gram_Matched'])), Long_Data['Matched_Keyphrase_Original_Spelling'], Long_Data[str(n+1) + 'gram'])     
        Long_Data = Long_Data.loc[~Long_Data['Matched_Keyphrase_Original_Spelling'].isin(excluded)]
        
    Long_Data = Long_Data.loc[Long_Data['Matched_Keyphrase'] != '', ] #filter away anything that doesn't match a keyphrase
    
    #Added 1/23 . . . an additional column for how the result will be displayed. For example, the keyphrases "TBI" and "Traumatic Brain Injury" will be displayed together as "Traumatic Brain Injury"
    Long_Data = Long_Data.merge(Keyphrases[['Keyphrase', 'Displayed_Keyphrase']], how = 'left', left_on = 'Matched_Keyphrase', right_on = 'Keyphrase')    
    
    #create columns of counts of each keyword    
    Keyphrase_Counts = Long_Data.groupby(['INDEX', 'Displayed_Keyphrase']).size().unstack(fill_value=0)
    
    for phrase in  displayed_keyphrase_list:
        if phrase not in list(Keyphrase_Counts):
            Keyphrase_Counts[phrase] = 0
    Keyphrase_Counts = Keyphrase_Counts[displayed_keyphrase_list]
    
    ############################################################################################################################################################################     
    #Clean up
    ############################################################################################################################################################################    
    Grouped_By_Keyphrase = Long_Data.groupby(['INDEX', 'Displayed_Keyphrase']).size().to_frame('count').reset_index()
     #create a colume that has the word, followed by ":", followed by its count
    Grouped_By_Keyphrase['Note Search Summary'] = Grouped_By_Keyphrase['Displayed_Keyphrase'] + ': ' + Grouped_By_Keyphrase['count'].astype(str)
    
    iLog_Search_Summary = pd.DataFrame(Grouped_By_Keyphrase.groupby('INDEX')['Note Search Summary'].apply(lambda x: ' | '.join(x)))
    
    #create column of summary of the original spellings of the words
    Grouped_By_Original_Spelling = Long_Data.groupby(['INDEX', 'Displayed_Keyphrase', 'Matched_Keyphrase_Original_Spelling']).size().to_frame('count').reset_index()
    Grouped_By_Original_Spelling['Original Spelling Summary'] = Grouped_By_Original_Spelling['Matched_Keyphrase_Original_Spelling'] + ': ' + Grouped_By_Original_Spelling['count'].astype(str)
    Grouped_By_Original_Spelling['Original Spelling'] = Grouped_By_Original_Spelling['Matched_Keyphrase_Original_Spelling']
    
    Display_Spelling = Grouped_By_Original_Spelling.groupby(['INDEX'])['Displayed_Keyphrase'].apply(lambda x : ','.join(x))
    
    Original_Spelling_Summary = Grouped_By_Original_Spelling.groupby(['INDEX', 'Displayed_Keyphrase'])['Original Spelling Summary'].apply(lambda x : ', '.join(x)).reset_index()
    Original_Spelling_Summary = pd.DataFrame(Original_Spelling_Summary.groupby('INDEX')['Original Spelling Summary'].apply(lambda x: ' | '.join(x)))
    
    Original_Spelling = Grouped_By_Original_Spelling.groupby(['INDEX', 'Displayed_Keyphrase'])['Original Spelling'].apply(lambda x : ','.join(x)).reset_index()
    Original_Spelling = pd.DataFrame(Original_Spelling.groupby('INDEX')['Original Spelling'].apply(lambda x: ','.join(x)))
    
    ############################################################################################################################################################################     
    # Combine everything into one table at claim unit level
    ############################################################################################################################################################################     
    Batch_Result = df[[level_key]].merge(Keyphrase_Counts, how = 'inner', left_index = True, right_index = True)
    Batch_Result = Batch_Result.merge(iLog_Search_Summary, how = 'inner', left_index = True, right_index = True)
    Batch_Result = Batch_Result.merge(Original_Spelling_Summary, how = 'inner', left_index = True, right_index = True)
    Batch_Result = Batch_Result.merge(Original_Spelling, how = 'inner', left_index = True, right_index = True)
    Batch_Result = Batch_Result.merge(Display_Spelling, how = 'inner', left_index = True, right_index = True)
    
    return Batch_Result
