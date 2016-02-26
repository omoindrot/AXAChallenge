# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from preprocessing import cleanup_data, lstm_data, split_train_val, lstm_test_set

from keras.utils.np_utils import accuracy
from keras.models import Graph
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

# Create a submission with a Fully Connected Network taking the week before as input

# X = pd.read_csv('data/train_2011_2012.csv', sep=';')
res = pd.read_csv('submission.txt', sep='\t')

# We can sort by date
# X.sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR'], inplace=True)

# Set of companies
companies_set = set(res['ASS_ASSIGNMENT'].value_counts().index)

# We only keep three columns: DATE, ASS_ASSIGNMENT, CSPL_CALLS
# X = pd.DataFrame({'DATE': X['DATE'], 'ASS_ASSIGNMENT': X['ASS_ASSIGNMENT'], 'CALLS': X['CSPL_CALLS']})

# Select the number of days to take into account
input_days = 10
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
model.add_input(name='input', input_shape=(input_days, 48))
model.add_node(LSTM(64), name='lstm', input='input')
model.add_node(Dropout(0.5), name='dropout', input='lstm')
model.add_node(Dense(48), name='prediction', input='dropout')
model.add_output(name='output', input='prediction')

model.compile('rmsprop', {'output': 'mse'})

print('Training...')
model.fit({'input': X_train, 'output': y_train},
          batch_size=16,
          nb_epoch=40, validation_data={'input': X_val, 'output': y_val})

predictions = model.predict({'input': X_val})['output']
MSE = np.mean((predictions-y_val)**2)
print "Mean Square Error of the model: ", MSE  # MSE = 0.80




# Submission
test_days = [date(2012, 1, 3), date(2012, 2, 8), date(2012, 3, 12), date(2012, 4, 16),
             date(2012, 5, 19), date(2012, 6, 18), date(2012, 7, 22), date(2012, 8, 21),
             date(2012, 9, 20), date(2012, 10, 24), date(2012, 11, 26), date(2012, 12, 28)]


X_test_companies = lstm_test_set(X, companies_set, test_days, input_days=10, flat=False)

y_predicted_companies = {}
for company in companies_set:
    y_predicted_companies[company] = model.predict({'input': X_test_companies[company]})['output']

# y_predicted_companies[company] of shape 12, 48

f1 = open('submission.txt', 'r')
submission = f1.readlines()
f1.close()

f2 = open('submission/lstm_10days_64hdim.txt', 'w')
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
    # company
    company = temp[1]
    # Get the data predicted
    num_line = -1
    for j in range(len(test_days)):
        if test_days[j] == res_date:
            num_line = j
    num_column = 2 * int(res_hour) + int(res_minutes)/30
    predicted = y_predicted_companies[company][num_line, num_column]
    #
    temp[2] = '%f' % predicted
    res = temp[0]+'\t'+temp[1]+'\t'+temp[2]+'\r\n'
    f2.write(res)

f2.close()
