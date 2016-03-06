# encoding: utf-8

import numpy as np

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


def model_creation(X_train, J_train, y_train, scalage, assignment_list, list_cod, input_days=4):
    model = {}
    MSE = {}

    nb_epoch = {'CAT': 1000, 'Domicile': 500, 'Gestion - Accueil Telephonique': 500, 'Japon': 300, 'Médical': 300,
                'Nuit': 300, 'RENAULT': 300, 'RTC': 300, 'Regulation Medicale': 300, 'SAP': 300, 'Services': 300,
                'Tech. Axa': 1000, 'Tech. Inter': 300, 'Tech. Total': 500, 'Téléphonie': 60}

    for assignment in assignment_list:
        cod_id = list_cod[assignment][0]
        size = X_train[assignment][cod_id][0].shape[1]
        #
        X_train_full = []
        J_train_full = []
        y_train_full = []
        for cod in X_train[assignment].keys():
            X_train_full += X_train[assignment][cod]
            J_train_full += J_train[assignment][cod]
            y_train_full += y_train[assignment][cod]
        #
        X_train_full = np.array(X_train_full)
        J_train_full = np.array(J_train_full)
        y_train_full = np.array(y_train_full)
        #
        model[assignment] = Graph()
        model[assignment].add_input(name='input', input_shape=(input_days, size))
        model[assignment].add_input(name='meteo', input_shape=(25,))
        model[assignment].add_node(LSTM(64), name='lstm', input='input')
        model[assignment].add_node(Dropout(0.3), name='dropout', input='lstm')
        # DONE: make the results always positive
        model[assignment].add_node(Dense(48), merge_mode='concat', name='prediction', inputs=['dropout', 'meteo'])
        model[assignment].add_node(Activation('relu'), name='relu', input='prediction')
        model[assignment].add_output(name='output', input='relu')
        model[assignment].compile('rmsprop', {'output': 'mse'})
        print('Training...')

        model[assignment].fit({'input': X_train_full, 'meteo': J_train_full, 'output': y_train_full}, batch_size=16,
                              nb_epoch=nb_epoch[assignment])
        #
        y_val = np.zeros((538, 48))
        y_predit = np.zeros((538, 48))
        for cod_id in X_train[assignment].keys():
            y_val += np.array(y_train[assignment][cod_id])
            y_pred_cod = model[assignment].predict({'input': np.array(X_train[assignment][cod_id]),
                                                    'meteo': np.array(J_train[assignment][cod_id])})['output']
            y_predit += y_pred_cod
        #
        y_val *= scalage[assignment]
        y_predit *= scalage[assignment]
        #
        MSE[assignment] = np.mean((y_predit-y_val)**2)
        print assignment
        print MSE[assignment]
        print

    return model, MSE
