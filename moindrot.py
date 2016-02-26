# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from preprocessing import lstm_train_data

from keras.utils.np_utils import accuracy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

# Here we create a simple submission, with the mean taken on all data
# Variants: different mean with day of the week, entreprise, time_slot

X = pd.read_csv('data/train_2011_2012.csv', sep=';')

res = pd.read_csv('submission.txt', sep='\t')

# We can sort by date
# X.sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR'], inplace=True)

# Set of companies
companies_set = set(res['ASS_ASSIGNMENT'].value_counts().index)

# We only keep three columns: DATE, ASS_ASSIGNMENT, CSPL_CALLS
X = pd.DataFrame({'DATE': X['DATE'], 'ASS_ASSIGNMENT': X['ASS_ASSIGNMENT'], 'CALLS': X['CSPL_CALLS']})


