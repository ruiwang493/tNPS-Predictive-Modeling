# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:27:51 2021

@author: JAMES.LI2
"""
import sys, os
import pandas as pd
import unittest
from pandas.testing import assert_series_equal

class TestNLPComplianceFunctions(unittest.TestCase):
    test_strings = ['james is living in los angeles (also known as la)',
                'There is nothing wrong with this phrase', 
                'james is going on vacation in italy!',
                "farmer's insurance", 
                'compliance text cleaning',
                'The date today is 7/30/2021']
    expected_strings = ['is living in also known as',
                        'there is nothing wrong with this phrase',
                        'is going on vacation in',
                        'farmers insurance',
                        'compliance text cleaning',
                        'the date today is']
    
    def testDataFrame(self):
        print("Executing Compliance Test of Series")
        # create series form a list
        test_series = pd.Series(self.test_strings)
        expected_series = pd.Series(self.expected_strings)
        
        assert_series_equal(expected_series,functions.clean_text_series_compliance(test_series))
    
    def testString(self):
        print("Executing test for a single string")
        
        for i in range(0, len(self.test_strings)):
            self.assertEqual(self.expected_strings[i],
                functions.clean_text_string_compliance(self.test_strings[i]))
            
    def testCustomRegex(self):
        print("Executing test for a custom regex")
        self.assertEqual('the date today is 7 30 2021',
                functions.clean_text_string_compliance('The date today is 7/30/2021', '[^A-Za-z0-9]+'))
    
    def testPass(self):
        print("Executing Default Test that Always Passes")

        self.assertEqual(1, 1, "This should pass")

if __name__ == '__main__':
    sys.path.insert(
        0, "C:\\Users\\" + os.getlogin() + "\\Desktop\\GitHub\\Claims-Code-Repo\\nlp_func"
    )
    
    import compliance_func as functions
    print("Beginning tests...")
    unittest.main()

