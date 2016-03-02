# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from preprocessing_cod import transform_data, build_training_set
from preprocessing import split_train_val
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
assignment_list =['CAT', 'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 'Gestion Amex',
                'Gestion Assurances', 'Gestion Clients', 'Gestion DZ', 'Gestion Relation Clienteles', 'Gestion Renault',
                'Japon', 'Manager', 'Mécanicien', 'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC',
                'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'Téléphonie']

num_companies = len(assignment_list)

# We only keep three columns: DATE, ASS_ASSIGNMENT, CSPL_CALLS
# X = pd.DataFrame({'DATE': X['DATE'], 'ASS_ASSIGNMENT': X['ASS_ASSIGNMENT'], 'CALLS': X['CSPL_CALLS']})
X_cleaned = pd.read_pickle('tmp/X_cod')

input_days = 5
X_ass, y_ass = build_training_set(X_cleaned, assignment_list, input_days=input_days, flat=True)
X_train_full = []
y_train_full = []
for assignment in assignment_list:
    X_train_full += X_ass[assignment]
    y_train_full += y_ass[assignment]
X_train, X_val, y_train, y_val = split_train_val(X_train_full, y_train_full, split_val=0.)

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
history = model.fit(X_train, y_train, batch_size=32, nb_epoch=10, validation_data=(X_val, y_val))

predictions = model.predict(X_val)
MSE = np.mean((predictions-y_val)**2)
print "Mean Square Error of the model: ", MSE  # MSE = 47.3 for input_days=4 and one hidden=256


# Testing the model on the sum for each ASS_ASSIGNMENT
num = len(y_ass['CMS'])
y_pred = {}
y_validation = {}
for assignment in assignment_list:
    temp_predicted = model.predict(np.array(X_ass[assignment]))
    temp_val = np.array(y_ass[assignment])
    y_pred[assignment] = np.zeros((num, 48))
    y_validation[assignment] = np.zeros((num, 48))
    for i in range(temp_predicted.shape[0]/num):
        y_pred[assignment] = y_pred[assignment] + temp_predicted[i*num: (i+1)*num]
        y_validation[assignment] = y_validation[assignment] + temp_val[i*num: (i+1)*num]

MSE = {}
MSE_tot = 0
for assignment in assignment_list:
    MSE = np.mean((y_pred[assignment]-y_validation[assignment])**2)
    print MSE, assignment
    MSE_tot += MSE

MSE_tot /= 27
MSE = np.mean((y_pred-y_validation)**2)
moy = np.mean(y_pred-y_validation)




# Compare prediction and y_val:
# i = 8
# i += 1
# for j in range(48):
#     print "%d - %f" % (y_val[i, j], predictions[i, j])
#     print
#
# print "local MSE: %f" % np.mean((predictions[i]-y_val[i])**2)

# Create submission

create_submission_cod(X_cleaned, model, assignment_list, input_days=input_days, flat=True)
