import pandas as pd
import numpy as np
from datetime import datetime

X = pd.read_csv('data/train_2011_2012.csv', sep=';')
Y = np.asarray(X)

res = pd.read_csv('submission.txt', sep='\t')
Y_res = np.asarray(res)

# set of type of companies 'ASS_ASSIGNMENT'
S = set(Y[:, 12])

# set of type of companies in the test set
S_test = set(Y_res[:, 1])


'''
Code for the cleanup of data
'''
X.describe()  # Describes each column
print X['SPLIT_COD'].value_counts()  # Histogram of a column
print X['CSPL_CALLS'].value_counts()

X = X.sort_values(by='DATE')

X_company = X[X.ASS_ASSIGNMENT == 'SAP']
print X[X.ASS_ASSIGNMENT.isin(['SAP', 'Tech. Axa'])]


# Drop any rows with missing data
X = X.dropna(how='any')

# Histogram of a column
print X['CSPL_CALLS'].value_counts()


'''
Drop the columns not needed
- because always only one value
- because we don't have it in the test set
'''
columns_dropped = ['DAYS_DS', 'ACD_COD', 'ACD_LIB', 'ASS_SOC_MERE', 'ASS_DIRECTORSHIP',
                   'ASS_PARTNER', 'ASS_POLE', 'ASS_BEGIN', 'ASS_END',
                   'CSPL_ABNCALLS1', 'CSPL_ABNCALLS2', 'CSPL_ABNCALLS3', 'CSPL_ABNCALLS4', 'CSPL_ABNCALLS5',
                   'CSPL_ABNCALLS6', 'CSPL_ABNCALLS7', 'CSPL_ABNCALLS8', 'CSPL_ABNCALLS9', 'CSPL_ABNCALLS10',
                   'CSPL_INTRVL', 'CSPL_INCOMPLETE']
X.drop(columns_dropped, 1, inplace=True)
X.drop('ASS_COMENT', 1, inplace=True)




# Example of element in X
print 'Example of a line in X (with largest phone calls number)'
print X.iloc[5391146].to_string()

