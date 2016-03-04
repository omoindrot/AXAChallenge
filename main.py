import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt


meteo = pd.read_pickle('tmp/meteo')
X_cleaned = pd.read_pickle('tmp/X_cod')

# days = pd.date_range('2011-01-01', '2011-06-30', freq='D')
# total = pd.DataFrame(np.zeros(731,), index=pd.date_range('2011-01-01', '2012-12-31', freq='D'))
#
# plt.figure(1)
# for cod_id in list_cod:
#     x = X_cleaned[cod_id].iloc[:, 27:].sum(axis=1)
#     total[0] += x.values
#     x.loc[days].plot()
#
# total.loc[days].plot()
# plt.show()

# List of cod for each assignment
assignment_list = ['CAT', 'CMS', 'Crises', 'Domicile', 'Gestion', 'Gestion - Accueil Telephonique', 'Gestion Amex',
                   'Gestion Assurances', 'Gestion Clients', 'Gestion DZ', 'Gestion Relation Clienteles', 'Gestion Renault',
                   'Japon', 'Manager', 'Mécanicien', 'Médical', 'Nuit', 'Prestataires', 'RENAULT', 'RTC',
                   'Regulation Medicale', 'SAP', 'Services', 'Tech. Axa', 'Tech. Inter', 'Tech. Total', 'Téléphonie']

list_cod = {}
for assignment in assignment_list:
    list_cod[assignment] = []

for cod_id in X_cleaned.keys():
    x = X_cleaned[cod_id]
    assignment = assignment_list[int(x.iloc[0, :27].argmax().split(' ')[1])]
    list_cod[assignment].append(cod_id)

total_days = pd.date_range('2011-01-01', '2012-12-31', freq='D')

leap_days = [date(2011, 1, 1), date(2011, 4, 9), date(2011, 5, 1), date(2011, 5, 8), date(2011, 6, 2),
             date(2011, 6, 13), date(2011, 7, 14), date(2011, 8, 15), date(2011, 11, 1), date(2011, 11, 11),
             date(2011, 12, 25),
             date(2012, 1, 1), date(2012, 4, 25), date(2012, 5, 1), date(2012, 5, 8), date(2012, 5, 17),
             date(2012, 5, 28), date(2012, 7, 14), date(2012, 8, 15), date(2012, 11, 1), date(2012, 11, 11),
             date(2012, 12, 25)]

X_bis = {}
for assignment in assignment_list:
    print 'assignment %d/%d' % (assignment_list.index(assignment), len(assignment_list))
    X_bis[assignment] = {}
    for cod_id in list_cod[assignment]:
        x = X_cleaned[cod_id]  # Dataframe of shape 731, 75 with an index on days
        for i in range(27):
            x.drop('assignment %d' % i, inplace=True)
        # Add year info
        x['y2011'] = 0.
        x['y2012'] = 0.
        for day in total_days:
            if day.year == 2011:
                x.loc[day]['y2011'] += 1.
            else:
                x.loc[day]['y2012'] += 1.
        # Add month info
        for i in range(1, 13):
            x['month%d' % i] = 0.
        for day in total_days:
            x.loc[day]['month%d' % day.month] += 1.
        # Add weekday info
        for i in range(7):
            x['weekday%d' % i] = 0.
        for day in total_days:
            x.loc[day]['weekday%d' % day.weekday()] += 1.
        # Add len(list_cod) columns of 0 / 1 for cod_id
        for i in range(len(list_cod[assignment])):
            x['cod%d' % i] = 0.
        x['cod%d' % list_cod[assignment].index(cod_id)] += 1.
        # Add the meteo data for 3 days ahead
        x['TEMP'] = 0.
        x['PRESSURE'] = 0.
        x['PRECIP'] = 0.
        for day in pd.date_range('2011-01-01', '2012-12-28', freq='D'):
            x.loc[day]['TEMP'] = meteo.loc[day]['TEMP']
            x.loc[day]['PRESSURE'] = meteo.loc[day]['PRESSURE']
            x.loc[day]['PRECIP'] = meteo.loc[day]['PRECIP']
        X_bis[assignment][cod_id] = x
        X_bis[assignment][cod_id].loc[:, 't0':'t47'] /= 20.
        X_bis[assignment][cod_id]['leap_day'] = 0.
        X_bis[assignment][cod_id]['leap_day'].loc[leap_days] = 1.


# pd.to_pickle(X_bis, 'tmp/X_bis')
X_bis = pd.read_pickle('tmp/X_bis')

# List of days in the test set
days_test = [date(2012, 1, 3), date(2012, 2, 8), date(2012, 3, 12), date(2012, 4, 16),
             date(2012, 5, 19), date(2012, 6, 18), date(2012, 7, 22), date(2012, 8, 21),
             date(2012, 9, 20), date(2012, 10, 24), date(2012, 11, 26), date(2012, 12, 28)]
input_days = 4

index_days = pd.date_range('2011-01-01', '2012-12-03', freq='D')
X_train = {}
J_train = {}
y_train = {}

for assignment in assignment_list:
    X_train[assignment] = {}
    J_train[assignment] = {}
    y_train[assignment] = {}
    for cod in list_cod[assignment]:
        X_train[assignment][cod] = []
        J_train[assignment][cod] = []
        y_train[assignment][cod] = []

count = 0
for assignment in assignment_list:
    count += 1
    print "index %d / %d" % (count, len(assignment_list))
    for cod_id in X_bis[assignment].keys():
        x = X_bis[assignment][cod_id]
        # Create the examples
        for day in index_days:
            valid = True
            for day_test in days_test:
                for i in range(5):
                    diff = (day_test - (day.date() + timedelta(7*i))).days
                    if 0 <= diff < 3:
                        valid = False
            if valid:
                days = pd.date_range(day, periods=4, freq='7D')
                train_example = x.loc[days].values
                train_j = x.loc[day+timedelta(28)]
                for i in range(len(X_bis[assignment].keys())):
                    train_j.drop('cod%d' % i, inplace=True)
                for i in range(48):
                    train_j.drop('t%d' % i, inplace=True)
                # for i in range(27):
                #     train_j.drop('assignment %d' % i, inplace=True)
                train_j = train_j.values  # size 25
                train_output = x.loc[day+timedelta(28)].loc['t0':'t47'].values
                X_train[assignment][cod_id].append(train_example)
                J_train[assignment][cod_id].append(train_j)
                y_train[assignment][cod_id].append(train_output)


from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


model = {}

for assignment in assignment_list:
    cod_id = list_cod[assignment][0]
    size = X_train[assignment][cod_id][0].shape[1]
    #
    X_train_full = []
    y_train_full = []
    for cod in X_train[assignment].keys():
        X_train_full += X_train[assignment][cod]
        y_train_full += y_train[assignment][cod]
    #
    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)
    #
    model[assignment] = Graph()
    model[assignment].add_input(name='input', input_shape=(input_days, size))
    model[assignment].add_input(name='meteo', input_shape=(25,))
    model[assignment].add_node(LSTM(128), name='lstm', input='input')
    model[assignment].add_node(Dropout(0.3), name='dropout', input='lstm')
    model[assignment].add_node(Dense(48), name='prediction', input='dropout')
    # DONE: make the results always positive
    model[assignment].add_node(Activation('relu'), name='relu', input='prediction')
    model[assignment].add_output(name='output', input='relu')
    model[assignment].add_node(Merge(inputs=['output', 'meteo'], mode='concat'))
    model[assignment].compile('rmsprop', {'output': 'mse'})
    print('Training...')
    model[assignment].fit({'input': X_train_full, 'output': y_train_full}, batch_size=16, nb_epoch=30)
    #
    y_val = np.zeros((538, 48))
    y_pred = np.zeros((538, 48))
    for cod_id in X_train[assignment].keys():
        y_val += np.array(y_train[assignment][cod_id])
        y_pred_cod = model[assignment].predict({'input': np.array(X_train[assignment][cod_id])})['output']
        y_pred += y_pred_cod
    #
    y_val *= 20.
    y_pred *= 20.
    #
    MSE = np.mean((y_pred-y_val)**2)
    print assignment
    print MSE
    print


# TEST TIME

X_test = {}
for assignment in assignment_list:
    X_test[assignment] = {}
    for cod in list_cod[assignment]:
        X_test[assignment][cod] = []

count = 0
for assignment in assignment_list:
    count += 1
    print "index %d / %d" % (count, len(assignment_list))
    for cod_id in X_bis[assignment].keys():
        x = X_bis[assignment][cod_id]
        # Create the test inputs
        for day in days_test:
            days = pd.date_range(end=day-timedelta(7), periods=4, freq='7D')
            train_example = x.loc[days].values
            X_test[assignment][cod_id].append(train_example)


y_pred = {}
for assignment in assignment_list:
    y_pred[assignment] = np.zeros((12, 48))
    for cod_id in X_train[assignment].keys():
        y_pred_cod = model[assignment].predict({'input': np.array(X_test[assignment][cod_id])})['output']
        y_pred[assignment] += y_pred_cod
    y_pred[assignment] *= 20


f1 = open('submission.txt', 'r')
submission = f1.readlines()
f1.close()

name = 'submission/submission_cod.txt'
f2 = open(name, 'w')
f2.write(submission[0])

for i in range(1, len(submission)):
    res = submission[i].split('\r')[0]
    temp = res.split('\t')
    #
    res_date = temp[0].split(' ')
    #
    res_time = res_date[1].split(':')
    res_hour = res_time[0]
    res_minutes = res_time[1]
    #
    res_date = res_date[0].split('-')
    res_date = date(int(res_date[0]), int(res_date[1]), int(res_date[2]))
    #
    # assignment
    assignment = temp[1]
    # Get the data predicted
    num_line = -1
    for j in range(len(days_test)):
        if days_test[j] == res_date:
            num_line = j
    num_column = 2 * int(res_hour) + int(res_minutes)/30
    predicted = y_pred[assignment][num_line, num_column]
    #
    temp[2] = '%f' % predicted
    res = temp[0]+'\t'+temp[1]+'\t'+temp[2]+'\r\n'
    f2.write(res)

f2.close()
