# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt


def clean_data(X, assignment_list, columns=None):
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


def transform_data(X_cleaned, meteo, assignment_list, leap_days):
    """
    Transform the data into a dictionary with each SPLIT_COD as key. The values are dataframes.
    :param X_cleaned: from previous function
    :param meteo: weather dataframe cleaned and normalized
    :param assignment_list: list of ASS_ASSIGNMENT
    :param leap_days: public holidays
    :return: dictionary of dataframes for each COD
    """
    list_cod = {}
    for assignment in assignment_list:
        list_cod[assignment] = []

    for cod_id in X_cleaned.keys():
        x = X_cleaned[cod_id]
        assignment = assignment_list[int(x.iloc[0, :27].argmax().split(' ')[1])]
        list_cod[assignment].append(cod_id)

    total_days = pd.date_range('2011-01-01', '2012-12-31', freq='D')



    scalage = {}
    for assignment in assignment_list:
        scalage[assignment] = 1.
        for cod_id in list_cod[assignment]:
            x = X_cleaned[cod_id]
            scalage[assignment] = max(x.loc[:, 't0':'t47'].max().max(), scalage[assignment])
        scalage[assignment] /= 3.

    X_bis = {}
    for assignment in assignment_list:
        print 'assignment %d/%d' % (assignment_list.index(assignment), len(assignment_list))
        X_bis[assignment] = {}
        for cod_id in list_cod[assignment]:
            x = X_cleaned[cod_id]  # Dataframe of shape 731, 75 with an index on days
            for i in range(27):
                x.drop('assignment %d' % i, axis=1, inplace=True)
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
            #
            x.loc[:, 't0':'t47'] /= scalage[assignment]
            x['leap_day'] = 0.
            x['leap_day'].loc[leap_days] = 1.
            X_bis[assignment][cod_id] = x

    pd.to_pickle((list_cod, X_bis, scalage), 'tmp/X_bis')
    return list_cod, X_bis, scalage


def build_training_set(X_bis, assignment_list, list_cod, days_test):
    """
    Build the training set
    :param X_cleaned: dictionary of dataframes for each split_cod
    :param assignment_list:
    :param input_days:
    :param flat:
    :return: dictionaries X_train, y_train
    """
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
                    train_j = np.zeros(25)
                    train_j[:-4] = x.loc[day+timedelta(28)].iloc[48:48+21].values
                    train_j[-4:] = x.loc[day+timedelta(28)].iloc[-4:].values
                    train_output = x.loc[day+timedelta(28)].loc['t0':'t47'].values
                    X_train[assignment][cod_id].append(train_example)
                    J_train[assignment][cod_id].append(train_j)
                    y_train[assignment][cod_id].append(train_output)

    return X_train, J_train, y_train


def build_test_set(X_bis, assignment_list, list_cod, days_test):

    X_test = {}
    J_test = {}
    for assignment in assignment_list:
        X_test[assignment] = {}
        J_test[assignment] = {}
        for cod in list_cod[assignment]:
            X_test[assignment][cod] = []
            J_test[assignment][cod] = []


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
                train_j = np.zeros(25)
                train_j[:-4] = x.loc[day].iloc[48:48+21].values
                train_j[-4:] = x.loc[day].iloc[-4:].values
                X_test[assignment][cod_id].append(train_example)
                J_test[assignment][cod_id].append(train_j)

    return X_test, J_test
