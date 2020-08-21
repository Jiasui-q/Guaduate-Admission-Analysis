"""
Runs all three parts of our research questions.
"""


import pandas as pd
from part1 import lasso_regression
from part2 import find_correlation
from part2 import boxplots_testscores_vs_admission
from part3 import university_rating_analysis


def main():
    """
    Loads in our data and then calls all functions that answer our research
    questions.
    """
    # Load in original data
    origin_data = pd.read_csv('/Users/apple/Desktop/CSE_163/cse163_project/'
                              + 'Admission_Predict_Ver1.1.csv',
                              sep=r'\s*,\s*', header=0, encoding='ascii',
                              engine='python')

    # Research question 1
    lasso_regression(origin_data)

    # Research question 2
    # We drop the 'Serial No.' column because it is unrelated to our analysis.
    df = origin_data.drop(columns=['Serial No.'])
    find_correlation(df)
    boxplots_testscores_vs_admission(df)

    # Research question 3
    university_rating_analysis(origin_data)


if __name__ == '__main__':
    main()
