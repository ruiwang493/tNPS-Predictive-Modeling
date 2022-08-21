# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:27:51 2021

@author: JAMES.LI2
"""
import sys, os
import pandas as pd
import unittest
from pandas.testing import assert_frame_equal

sys.path.insert(
    0, "C:\\Users\\" + os.getlogin() + "\\Desktop\\GitHub\\Claims-Code-Repo\\nlp_func"
)
import compliance_func as comp_func
import fuzzy_word_search_func as fuzzy_func

def assert_frame_not_equal(*args, **kwargs):
    try:
        assert_frame_equal(*args, **kwargs)
    except AssertionError:
        # frames are not equal
        pass
    else:
        # frames are equal
        raise AssertionError

class TestNLPComplianceFunctions(unittest.TestCase):    
    df = pd.read_csv("C:\\Users\\" + os.getlogin() + "\\Desktop\\GitHub\\Claims-Code-Repo\\nlp_func\\unit_test\\test_notes.csv")
    text_field = 'Review'
    level_key = 'ID'
    df[text_field] = comp_func.clean_text_series_compliance(df[text_field])

    Keyphrases = pd.read_csv("C:\\Users\\" + os.getlogin() + '\\Desktop\\GitHub\\Claims-Code-Repo\\nlp_func\\unit_test\\sample_keyphrases.csv')

    def testSearchFuzzyFuzzRatio(self):
        print("Executing Test showing Fuzz Ratio makes matching correctly")
        # create series form a list
        output1 = fuzzy_func.FuzzyWordSearch(self.df, self.text_field, self.Keyphrases, self.level_key, list(), fuzz_ratio = 80)
        temp_df = self.df.copy()
        temp_df[self.text_field] = temp_df[self.text_field].str.replace(' movie ', ' movei ')
        output2 = fuzzy_func.FuzzyWordSearch(temp_df, self.text_field, self.Keyphrases, self.level_key, list(), fuzz_ratio = 80)
        
        expected_column_diff = ['Original Spelling', 'Original Spelling Summary']
        assert_frame_equal(output1.drop(columns = expected_column_diff), output2.drop(columns = expected_column_diff))
        assert_frame_not_equal(output1[expected_column_diff], output2[expected_column_diff])
   
    def testSearchFuzzyRatioFailed(self):
        print("Executing Test showing Fuzz Ratio of 88 does not make movie and movei end up correctly")
        # create series form a list
        output1 = fuzzy_func.FuzzyWordSearch(self.df, self.text_field, self.Keyphrases, self.level_key, list())
        temp_df = self.df.copy()
        temp_df[self.text_field] = temp_df[self.text_field].str.replace(' movie ', ' movei ')
        output2 = fuzzy_func.FuzzyWordSearch(temp_df, self.text_field, self.Keyphrases, self.level_key, list())
        
        expected_column_diff = ['Original Spelling', 'Original Spelling Summary']
        assert_frame_not_equal(output1.drop(columns = expected_column_diff), output2.drop(columns = expected_column_diff))

    def testSearchFuzzyExcluded(self):
        print("Executing Test showing that an excluded word will be omitted")
        output1 = fuzzy_func.FuzzyWordSearch(self.df, self.text_field, self.Keyphrases, self.level_key, list())
        self.assertGreaterEqual(output1['Original Spelling'].str.contains('movies').sum(), 1)
        output2 = fuzzy_func.FuzzyWordSearch(self.df, self.text_field, self.Keyphrases, self.level_key, ['movies'])
        self.assertEqual(output2['Original Spelling'].str.contains('movies').sum(), 0)
        
    def testPass(self):
        print("Executing Default Test that Always Passes")

        self.assertEqual(1, 1, "This should pass")

if __name__ == '__main__':    
    print("Beginning tests...")
    unittest.main()

