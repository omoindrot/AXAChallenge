# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from preprocessing import clean_data, transform_data, build_training_set, build_test_set
from LSTM import model_creation
from submission import create_submission


# List of assignments in the test set
assignment_list = ['CAT', 'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 'Gestion Amex',
                   'Gestion Assurances', 'Gestion Clients', 'Gestion DZ', 'Gestion Relation Clienteles', 'Gestion Renault',
                   'Japon', 'Manager', 'Mécanicien', 'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC',
                   'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'Téléphonie']

leap_days = [date(2011, 1, 1), date(2011, 4, 9), date(2011, 5, 1), date(2011, 5, 8), date(2011, 6, 2),
             date(2011, 6, 13), date(2011, 7, 14), date(2011, 8, 15), date(2011, 11, 1), date(2011, 11, 11),
             date(2011, 12, 25),
             date(2012, 1, 1), date(2012, 4, 25), date(2012, 5, 1), date(2012, 5, 8), date(2012, 5, 17),
             date(2012, 5, 28), date(2012, 7, 14), date(2012, 8, 15), date(2012, 11, 1), date(2012, 11, 11),
             date(2012, 12, 25)]

# List of days in the test set
days_test = [date(2012, 1, 3), date(2012, 2, 8), date(2012, 3, 12), date(2012, 4, 16),
             date(2012, 5, 19), date(2012, 6, 18), date(2012, 7, 22), date(2012, 8, 21),
             date(2012, 9, 20), date(2012, 10, 24), date(2012, 11, 26), date(2012, 12, 28)]

# Number of days to take in the input
input_days = 4

# We get the weather produced by file meteo.py
meteo = pd.read_pickle('tmp/meteo')


# WE GET THE CLEANED DATA
# X = pd.read_csv('data/train_2011_2012.csv', sep=';')
# X_cleaned = clean_data(X, assignment_list)

# X_cleaned was saved in pickle format so it is easier to take it there
X_cleaned = pd.read_pickle('tmp/X_cod')


# WE GET THE TRANSFORMED DATA
list_cod, X_bis = transform_data(X_cleaned, meteo, assignment_list, leap_days)

list_cod, X_bis, scalage = pd.read_pickle('tmp/X_bis')


# WE CREATE THE TRAINING SET

# X_train, J_train, y_train = build_training_set(X_bis, assignment_list, list_cod, days_test)
X_train, J_train, y_train = pd.read_pickle('tmp/training_set')


# WE TRAIN THE MODEL
# returns a dictionary of models for each ASSIGNMENT
model, MSE = model_creation(X_train, J_train, y_train, scalage, assignment_list, list_cod, input_days=4)


# TEST TIME
# We first get the test set
X_test, J_test = build_test_set(X_bis, assignment_list, list_cod, days_test)

# We use the models to predict the output y_pred
y_pred = pd.read_pickle('tmp/y_pred2')
# y_pred = {}
for assignment in assignment_list:
    y_pred[assignment] = np.zeros((12, 48))
    for cod_id in X_train[assignment].keys():
        y_pred_cod = model[assignment].predict({'input': np.array(X_test[assignment][cod_id]), 'meteo': np.array(J_test[assignment][cod_id])})['output']
        y_pred[assignment] += y_pred_cod

    y_pred[assignment] *= scalage[assignment]


# We can save the output for future use
pd.to_pickle(y_pred, 'tmp/y_pred3')
# y_pred = pd.read_pickle('tmp/y_pred2')


# SUBMISSION
name = 'submission/test.txt'
create_submission(name, y_pred, days_test)

