# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from preprocessing import cleanup_data, lstm_data, lstm_data_assignment, split_train_val
from submission import create_submission

from keras.utils.np_utils import accuracy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

# Create a submission with a Fully Connected Network taking the week before as input

# X = pd.read_csv('data/train_2011_2012.csv', sep=';')

# res = pd.read_csv('submission.txt', sep='\t')

# We can sort by date
# X.sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR'], inplace=True)

# List of companies
assignment_list = ['CAT', 'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 'Gestion Amex',
                   'Gestion Assurances', 'Gestion Clients', 'Gestion DZ', 'Gestion Relation Clienteles', 'Gestion Renault',
                   'Japon', 'Manager', 'Mécanicien', 'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC',
                   'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'Téléphonie']

num_companies = len(assignment_list)

# We only keep three columns: DATE, ASS_ASSIGNMENT, CSPL_CALLS
# X = pd.DataFrame({'DATE': X['DATE'], 'ASS_ASSIGNMENT': X['ASS_ASSIGNMENT'], 'CALLS': X['CSPL_CALLS']})
X_cleaned = pd.read_pickle('tmp/X_cleaned')

input_days = 5
X_train, y_train = lstm_data(X_cleaned, assignment_list, input_days=input_days, flat=True)
X_train, X_val, y_train, y_val = split_train_val(X_train, y_train)

print '-'*50
print '%d training examples of size (%d, )' % X_train.shape
print '%d training outputs of size (%d, )' % y_train.shape
print '-'*50
print '%d validation examples of size (%d, )' % X_val.shape
print '%d validation outputs of size (%d, )' % y_val.shape

model = Sequential()
model.add(Dense(256, input_shape=(input_days*(num_companies+48),)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(48))
model.add(Activation('relu'))

model.compile(loss='mse', optimizer='rmsprop')

print('Training...')
history = model.fit(X_train, y_train, batch_size=8, nb_epoch=100, validation_data=(X_val, y_val))

predictions = model.predict(X_val)
MSE = np.mean((predictions-y_val)**2)
print "Mean Square Error of the model: ", MSE  # MSE = 47.3 for input_days=4 and one hidden=256

# Compare prediction and y_val:
# i = 8
# i += 1
# for j in range(48):
#     print "%d - %f" % (y_val[i, j], predictions[i, j])
#     print
#
# print "local MSE: %f" % np.mean((predictions[i]-y_val[i])**2)

# Create submission

create_submission(X_cleaned, model, assignment_list, input_days=input_days, flat=True)
