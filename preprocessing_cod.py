# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt


def transform_data(X, assignment_list, columns=None):
    """
    Transform the initial DataFrame X to a more appropriate DataFrame.
    For instance, string columns are transformed into one hot vectors
    :param X: DataFrame from the csv file
    :param assignment_list: set of assignments of interest (i.e. ASS_ASSIGNMENT of interest)
    :return: X_transformed, new DataFrame with columns DATE, multiple feature columns (float32) and CSPL_CALLS
    """
    X_transformed = pd.DataFrame()
    # DATE: dates in 2011 and 2012
    # SPLIT_COD: unique attribute of the data center
    # ASS_... : information concerning the rest
    if columns is None:
        columns = ['DATE', 'SPLIT_COD', 'ASS_SOC_MERE', 'ASS_DIRECTORSHIP', 'ASS_ASSIGNMENT', 'ASS_PARTNER', 'ASS_POLE', 'CSPL_CALLS']
    for column in columns:
        X_transformed[column] = X[column]

    # We only take the inputs where ASS_ASSIGNMENT is in the test set
    X_list = []
    for assignment in assignment_list:
        X_list.append(X_transformed[X_transformed['ASS_ASSIGNMENT'] == assignment])
    X_transformed = pd.concat(X_list)

    # There are 466 different SPLIT_COD now
    X_cleaned = {}

    # http://pandas.pydata.org/pandas-docs/stable/groupby.html
    index_cod = list(X_transformed['SPLIT_COD'].value_counts().index)
    index_cod.sort()
    index_days = pd.date_range('2011-01-01 00:00:00', periods=2*365+1, freq='D')
    index = pd.date_range('2011-01-01 00:00:00', periods=(2*365+1)*24*2, freq='30MIN')
    begin = date(2011, 1, 1)
    count = 0

    # We initialize the dictionary X_cleaned
    for i in index_cod:
        count += 1
        print "index %d / %d" % (count, len(index_cod))
        x = X_transformed[X_transformed['SPLIT_COD'] == i]
        assignment = x['ASS_ASSIGNMENT'].iloc[0]
        x_cleaned = pd.DataFrame(index=index_days)
        for j in range(len(assignment_list)):
            if j == assignment_list.index(assignment):
                x_cleaned['assignment %d' % j] = 1.
                print j
            else:
                x_cleaned['assignment %d' % j] = 0.
        for j in range(48):
            x_cleaned['t%d' % j] = 0.
        X_cleaned[i] = x_cleaned

    # Now we fill in the values
    X_grouped = X_transformed.groupby(['SPLIT_COD'])
    count = 0
    for cod_id, x in X_grouped:
        count += 1
        print "index %d / %d" % (count, len(index_cod))
        assignment = x['ASS_ASSIGNMENT'].iloc[0]
        calls = x.groupby('DATE').sum()['CSPL_CALLS']
        for slot in index:
            pos_i = slot.date()
            pos_j = 't%d' % (2*slot.hour + slot.minute/30)
            if slot.isoformat(' ')+'.000' in calls.index:
                X_cleaned[cod_id].loc[pos_i, pos_j] += calls[slot.isoformat(' ')+'.000']

    # We got a dictionary of DataFrames X_cleaned with days_index and 27 columns for assignment and 48 for calls
    # We save this to 'tmp/X_cod' with pickle
    # pd.to_pickle(X_cleaned, 'tmp/X_cod')
    return X_cleaned


def build_training_set(X_cleaned, assignment_list, input_days=5, flat=False):
    """
    Build the training set
    :param X_cleaned: dictionary of dataframes for each split_cod
    :param assignment_list:
    :param input_days:
    :param flat:
    :return: lists X_train, y_train
    """

    # List of days in the test set
    days_test = [date(2012, 1, 3), date(2012, 2, 8), date(2012, 3, 12), date(2012, 4, 16),
                 date(2012, 5, 19), date(2012, 6, 18), date(2012, 7, 22), date(2012, 8, 21),
                 date(2012, 9, 20), date(2012, 10, 24), date(2012, 11, 26), date(2012, 12, 28)]
    index_days = pd.date_range('2011-01-01 00:00:00', periods=2*365+1-(input_days+3)+1, freq='D')
    X_train = {}
    y_train = {}
    for assignment in assignment_list:
        X_train[assignment] = []
        y_train[assignment] = []

    count = 0
    for cod_id in X_cleaned.keys():
        count += 1
        print "index %d / %d" % (count, len(X_cleaned.keys()))
        x = X_cleaned[cod_id]
        # Get the ASS_ASSIGNMENT
        assignment = assignment_list[int(x.iloc[0, :27].argmax().split(' ')[1])]
        # Create the examples
        for day in index_days:
            valid = True
            for day_test in days_test:
                diff = (day_test - day.date()).days
                if 0 <= diff < input_days+3:
                    valid = False
            if valid:
                train_example = x.loc[day:day + timedelta(input_days-1)].values
                train_output = x.loc[day + timedelta(input_days-1+3)].iloc[len(assignment_list):].values
                if not flat:
                    X_train[assignment].append(train_example)
                    y_train[assignment].append(train_output)
                else:
                    X_train[assignment].append(train_example.reshape(input_days*(48+len(assignment_list))))
                    y_train[assignment].append(train_output.reshape(48))

    return X_train, y_train


def build_test_set(X_cleaned, assignment_list, input_days=5, flat=False):
    """
    Build the training set
    :param X_cleaned: dictionary of dataframes for each split_cod
    :param assignment_list:
    :param input_days:
    :param flat:
    :return: lists X_train, y_train
    """

    # List of days in the test set
    days_test = [date(2012, 1, 3), date(2012, 2, 8), date(2012, 3, 12), date(2012, 4, 16),
                 date(2012, 5, 19), date(2012, 6, 18), date(2012, 7, 22), date(2012, 8, 21),
                 date(2012, 9, 20), date(2012, 10, 24), date(2012, 11, 26), date(2012, 12, 28)]
    index_days = pd.date_range('2011-01-01 00:00:00', periods=2*365+1-(input_days+3)+1, freq='D')

    # Dictionary for each ASS_ASSIGNMENT, with arrays X_test inside
    X_test_assignments = {}

    count = 0
    for cod_id in X_cleaned.keys():
        count += 1
        print "index %d / %d" % (count, len(X_cleaned.keys()))
        x = X_cleaned[cod_id]
        for day in index_days:
            valid = True
            for day_test in days_test:
                diff = (day_test - day.date()).days
                if 0 <= diff < input_days+3:
                    valid = False
            if valid:
                test_example = x.loc[day:day + timedelta(input_days-1)].values
                if not flat:
                    X_test.append(test_example)
                else:
                    X_test.append(test_example.reshape(input_days*(48+len(assignment_list))))

    return X_test



def create_submission_cod(X_cleaned, model, companies_set, input_days=4, flat=False):
    test_days = [date(2012, 1, 3), date(2012, 2, 8), date(2012, 3, 12), date(2012, 4, 16),
                 date(2012, 5, 19), date(2012, 6, 18), date(2012, 7, 22), date(2012, 8, 21),
                 date(2012, 9, 20), date(2012, 10, 24), date(2012, 11, 26), date(2012, 12, 28)]

    X_test_companies = build_test_set(X_cleaned, companies_set, test_days, input_days=input_days, flat=flat)

    y_predicted_companies = {}
    for company in companies_set:
        if not flat:
            y_predicted_companies[company] = model.predict({'input': X_test_companies[company]})['output']
            y_predicted_companies[company][y_predicted_companies[company] < 0] = 0.
        else:
            y_predicted_companies[company] = model.predict(X_test_companies[company])
            y_predicted_companies[company][y_predicted_companies[company] < 0] = 0.

    # y_predicted_companies[company] of shape 12, 48

    f1 = open('submission.txt', 'r')
    submission = f1.readlines()
    f1.close()

    if not flat:
        name = 'submission/lstm_%ddays_64hdim.txt' % input_days
    else:
        name = 'submission/nn_%ddays_256hdim.txt' % input_days

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
