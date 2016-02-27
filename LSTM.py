# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from preprocessing import cleanup_data, lstm_data, split_train_val, lstm_test_set
from submission import create_submission

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

# Create a submission with a Fully Connected Network taking the week before as input

# X = pd.read_csv('data/train_2011_2012.csv', sep=';')
# res = pd.read_csv('submission.txt', sep='\t')

# We can sort by date
# X.sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR'], inplace=True)

# Set of companies
companies_set =['CAT', 'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 'Gestion Amex',
                'Gestion Assurances', 'Gestion Clients', 'Gestion DZ', 'Gestion Relation Clienteles', 'Gestion Renault',
                'Japon', 'Manager', 'Mécanicien', 'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC',
                'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'Téléphonie']

num_companies = len(companies_set)

# We only keep three columns: DATE, ASS_ASSIGNMENT, CSPL_CALLS
# X = pd.DataFrame({'DATE': X['DATE'], 'ASS_ASSIGNMENT': X['ASS_ASSIGNMENT'], 'CALLS': X['CSPL_CALLS']})

# Select the number of days to take into account
input_days = 4
# X_cleaned = cleanup_data(X, companies_set)
X_cleaned = pd.read_pickle('tmp/X_cleaned')

X_train, y_train = lstm_data(X_cleaned, companies_set, input_days=input_days, flat=False)
X_train, X_val, y_train, y_val = split_train_val(X_train, y_train)

print '-'*50
print '%d training examples of size (%d, %d)' % X_train.shape
print '%d training outputs of size (%d,)' % y_train.shape
print '-'*50
print '%d validation examples of size (%d, %d)' % X_val.shape
print '%d validation outputs of size (%d, )' % y_val.shape

model = Graph()
model.add_input(name='input', input_shape=(input_days, num_companies+48))
model.add_node(LSTM(64), name='lstm', input='input')
model.add_node(Dropout(0.5), name='dropout', input='lstm')
model.add_node(Dense(48), name='prediction', input='dropout')
# TODO: make the results always positive
model.add_node(Activation('relu'), name='relu', input='prediction')
model.add_output(name='output', input='relu')

model.compile('rmsprop', {'output': 'mse'})

print('Training...')
model.fit({'input': X_train, 'output': y_train},
          batch_size=16,
          nb_epoch=20, validation_data={'input': X_val, 'output': y_val})


y_predicted = model.predict({'input': X_val})['output']
y_predicted[y_predicted<0] = 0.
MSE = np.mean((y_predicted-y_val)**2)
# MSE = 67.65 (with full data, and input_days=4)

create_submission(X_cleaned, model, companies_set, input_days=input_days)
