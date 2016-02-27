# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from preprocessing import cleanup_data, lstm_data, split_train_val, lstm_test_set

from keras.utils.np_utils import accuracy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

# Create a submission with a Fully Connected Network taking the week before as input

# X = pd.read_csv('data/train_2011_2012.csv', sep=';')

res = pd.read_csv('submission.txt', sep='\t')

# We can sort by date
# X.sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR'], inplace=True)

# Set of companies
companies_set = set(res['ASS_ASSIGNMENT'].value_counts().index)

# We only keep three columns: DATE, ASS_ASSIGNMENT, CSPL_CALLS
# X = pd.DataFrame({'DATE': X['DATE'], 'ASS_ASSIGNMENT': X['ASS_ASSIGNMENT'], 'CALLS': X['CSPL_CALLS']})
X_cleaned = pd.read_pickle('tmp/X_cleaned')

input_days = 3
X_train, y_train = lstm_data(X_cleaned, companies_set, input_days=input_days, flat=True)
X_train, X_val, y_train, y_val = split_train_val(X_train, y_train)

print '-'*50
print '%d training examples of size (%d, )' % X_train.shape
print '%d training outputs of size (%d, )' % y_train.shape
print '-'*50
print '%d validation examples of size (%d, )' % X_val.shape
print '%d validation outputs of size (%d, )' % y_val.shape

model = Sequential()
model.add(Dense(256, input_shape=(input_days*48,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(48))

model.compile(loss='mse', optimizer='rmsprop')

print('Training...')
model.fit(X_train, y_train, batch_size=8, nb_epoch=10,
          validation_data=(X_val, y_val))

predictions = model.predict(X_val)
MSE = np.mean((predictions-y_val)**2)
print "Mean Square Error of the model: ", MSE  # MSE = 84.8 for input_days=3
