# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta


# We only keep three columns: DATE, ASS_ASSIGNMENT, CSPL_CALLS
# X = pd.DataFrame({'DATE': X['DATE'], 'ASS_ASSIGNMENT': X['ASS_ASSIGNMENT'], 'CALLS': X['CSPL_CALLS']})


def cleanup_data(X, assignment_list):
    """
    Clean the data to create a dataframe of size (num_assignments, 365*2+1, 48).
    Each assignment has every day available, with 48 slots
    :param X: DataFrame with columns DATE, ASS_ASSIGNMENT and CALLS
    :param assignment_list: assignments for which we create the cleaned data
    :return: X_cleaned, DataFrame withe columns DATE, ASS_ASSIGNMENT and 48 columns CALLS
    """
    X_grouped = X.groupby(['ASS_ASSIGNMENT', 'DATE']).sum()
    index_days = pd.date_range('2011-01-01 00:00:00', periods=2*365+1, freq='D')
    index = pd.date_range('2011-01-01 00:00:00', periods=(2*365+1)*24*2, freq='30MIN')
    begin = date(2011, 1, 1)

    X_cleaned = pd.DataFrame()
    count = 0
    for assignment in assignment_list:
        count += 1
        print "Company %s (%d / %d)" % (assignment, count, len(assignment_list))
        X_assignment_grouped = X_grouped.loc[assignment]
        X_assignment_cleaned = pd.DataFrame({'DAY': index_days, 'ASS_ASSIGNMENT': assignment})
        for i in range(48):
            X_assignment_cleaned[i] = 0.
        # For each slot we add the calls
        for slot in index:
            i = (slot.date() - begin).days  # index in values to insert (0<=i<731)
            j = 2 + 2*slot.hour + slot.minute/30  # index in values to insert (0<=j<48)
            if slot.isoformat(' ')+'.000' in X_assignment_grouped.index:
                X_assignment_cleaned.iloc[i, j] += X_grouped.loc[assignment, slot.isoformat(' ')+'.000']['CALLS']
    # We concatenate the DataFrame created
        X_cleaned = pd.concat([X_cleaned, X_assignment_cleaned])

    return X_cleaned


def lstm_data_assignment(X_cleaned, assignment_list, assignment, input_days=4, flat=False):
    """
    Creates a training dataset with (input_days, len(assignment_list)+48) in, and 48 out
    :param X_cleaned: DataFrame with 50 columns (ASS_ASSIGNMENT, DATE, 48 slots)
    :param assignment: name of the assignment for which we create the dataset
    :param assignment_list: names of the assignments in total
    :param input_days: number of days to take into account
    :param flat: if you want the shape of each row to be (input_days*(48+len),)

    :return: lists X_train, y_train of same length
    """

    X_assignment_cleaned = X_cleaned[X_cleaned['ASS_ASSIGNMENT'] == assignment]
    index_assignment = assignment_list.index(assignment)

    X_assignment_cleaned = X_assignment_cleaned.iloc[:, 2:]

    # List of days in the test set
    days_test = [date(2012, 1, 3), date(2012, 2, 8), date(2012, 3, 12), date(2012, 4, 16),
                 date(2012, 5, 19), date(2012, 6, 18), date(2012, 7, 22), date(2012, 8, 21),
                 date(2012, 9, 20), date(2012, 10, 24), date(2012, 11, 26), date(2012, 12, 28)]
    begin = date(2011, 1, 1)

    X_train = []
    y_train = []
    for i in range(2*365+1-(input_days+3)+1):
        valid = True
        for day in days_test:
            diff = (day-begin).days - i
            if 0 <= diff < input_days+3:
                valid = False
        if valid:
            train_example = np.zeros((input_days, 48+len(assignment_list)))
            train_example[:, len(assignment_list):] += X_assignment_cleaned[i:i+input_days].values
            train_example[:, index_assignment] += 1.
            train_output = X_assignment_cleaned.iloc[i+input_days+2].values
            if not flat:
                X_train.append(train_example)
                y_train.append(train_output)
            else:
                X_train.append(train_example.reshape(input_days*(48+len(assignment_list))))
                y_train.append(train_output.reshape(48))
    return X_train, y_train


def lstm_data(X_cleaned, assignment_list, input_days=4, flat=False):
    """
    Creates a dataset for all the assignments
    :param X_cleaned: DataFrame with 50 columns (ASS_ASSIGNMENT, DATE, 48 slots)
    :param assignment_list: names of the assignments for which we create the dataset
    :param input_days: number of days to take into account
    :param flat: if you want the shape of each row to be (input_days*48,)
    :return: lists X_train, y_train of same length
    """

    X_train = []
    y_train = []
    count = 0
    for assignment in assignment_list:
        count += 1
        print "Company in progress: %s (%d/%d)" % (assignment, count, len(assignment_list))
        x, y = lstm_data_assignment(X_cleaned, assignment_list, assignment, input_days=input_days, flat=flat)
        X_train += x
        y_train += y
    return X_train, y_train


def split_train_val(X, y, split_val=0.2):
    """
    Splits lists between training and validation set
    :param X: list of training examples
    :param y: list of outputs, same length as X
    :param split_val: share of inputs used as validation set

    :return: X_train, X_val, y_train, y_val numpy arrays
    """
    X_train = np.array(X)
    y_train = np.array(y)

    day_indexes = np.arange(len(X_train))
    np.random.shuffle(day_indexes)

    train_indexes = day_indexes[split_val*len(X_train):]
    val_indexes = day_indexes[:split_val*len(X_train)]

    X_val = X_train[val_indexes]
    X_train = X_train[train_indexes]

    y_val = y_train[val_indexes]
    y_train = y_train[train_indexes]

    return X_train, X_val, y_train, y_val


def lstm_test_set(X_cleaned, assignment_list, test_days, input_days=4, flat=False):
    """
    Creates the test dataset with (input_days, 48) in, and 48 out
    :param X_cleaned: DataFrame with 50 columns (ASS_ASSIGNMENT, DATE, 48 slots)
    :param assignment_list: names of the assignments for which we create the dataset
    :param test_days: days trying to be predicted
    :param input_days: number of days to take into account
    :param flat: if you want the shape of each row to be (input_days*48,)

    :return: lists X_test
    """

    begin = date(2011, 1, 1)
    X_test_assignments = {}
    for assignment in assignment_list:
        X_test_assignments[assignment] = []
        X_cleaned_assignment = X_cleaned[X_cleaned['ASS_ASSIGNMENT'] == assignment].iloc[:, 2:]
        index_assignment = assignment_list.index(assignment)
        for day in test_days:
            i = (day - begin).days - 3 - input_days + 1
            test_example = np.zeros((input_days, (len(assignment_list)+48)))
            test_example[:, len(assignment_list):] = X_cleaned_assignment[i: i+input_days].values
            test_example[:, index_assignment] = 1.
            if not flat:
                X_test_assignments[assignment].append(test_example)
            else:
                X_test_assignments[assignment].append(test_example.reshape(input_days*(len(assignment_list)+48)))
        X_test_assignments[assignment] = np.array(X_test_assignments[assignment])

    return X_test_assignments



'''
If you want to create columns YEAR...
# We create columns YEAR, MONTH, DATE, HOUR
X['DATE'] = X['DATE'].str.split(' ')
X['YEAR'] = X['DATE'].apply(lambda s: s[0])
X['TIME'] = X['DATE'].apply(lambda s: s[1])
X['YEAR'] = X['YEAR'].str.split('-')
X['MONTH'] = X['YEAR'].apply(lambda s: int(s[1]))
X['DAY'] = X['YEAR'].apply(lambda s: int(s[2]))
X['YEAR'] = X['YEAR'].apply(lambda s: int(s[0]))
X['TIME'] = X['TIME'].str.split(':')
X['HOUR'] = X['TIME'].apply(lambda s: int(s[0]))
X['MINUTE'] = X['TIME'].apply(lambda s: s[1])
X['MINUTE'] = X['MINUTE'].apply(lambda s: int(s)/60.)
X['HOUR'] += X['MINUTE']
#X.drop('DATE', axis=1, inplace=True)
X.drop('TIME', axis=1, inplace=True)
X.drop('MINUTE', axis=1, inplace=True)
'''


# List of assignments ('ASS_ASSIGNMENT')
# X['ASS_ASSIGNMENT'].value_counts()
'''
Total: 55 valeurs différentes
Total dans le test set: 27 entreprises différentes

Téléphonie                        2931003
#A DEFINIR                         1050065
Médical                            790971
#Technique Belgique                 495548
#Technique International            304851
RENAULT                            289182
Tech. Axa                          282303
Tech. Inter                        265290
Services                           264384
#TPA                                220747
Nuit                               161517
Tech. Total                        137288
Domicile                           133187
#TAI - RISQUE                       127610
#Medicine                           121249
#Technical                          121159
#LifeStyle                          121112
#Maroc - Renault                    116266
#TAI - SERVICE                      115190
#TAI - CARTES                        92063
#TAI - RISQUE SERVICES               88352
Gestion - Accueil Telephonique      85346
Manager                             69928
Japon                               66601
Regulation Medicale                 63628
#Finances PCX                        42863
Mécanicien                          36586
#KPT                                 34751
#Maroc - Génériques                  33457
Gestion                             33035
SAP                                 32782
#TAI - PNEUMATIQUES                  32549
Gestion Renault                     31490
RTC                                 30407
CAT                                 29921
#AEVA                                29449
Gestion Assurances                  27502
#NL Technique                        24551
Crises                              24508
#Truck Assistance                    24403
Gestion Clients                     24142
#NL Médical                          23869
#Divers                              23580
Gestion DZ                          22641
Gestion Relation Clienteles         20402
#TAI - PANNE MECANIQUE               13756
Gestion Amex                        13248
#FO Remboursement                    13198
#Réception                           12846
CMS                                 12574
#DOMISERVE                           11395
Prestataires                         9598
#IPA Belgique - E/A MAJ               7894
#Evenements                            431
#Juridique                              12
'''



# List of companies ('ASS_SOC_MERE')
# X['ASS_SOC_MERE'].value_counts()
'''
Total: 8 valeurs différentes
Total dans le test set: 27 entreprises différentes

Entity1 France                        5348091 Médical has 249167 inputs
Entity2 Belgique                      2257811 (only in Médical, with 541804 inputs)
#TAI                                    974348
#Entity2 Suisse                         421863
#Entity1 Maroc Services                 116266
#Entity1 Maroc                           33457
#AMERICAN EXPRESS VOYAGES D'AFFAIRE      29449
#DOMISERVE                               11395
'''

'''
ASS_SOC_MERE     - ASS_DIRECTORSHIP - ASS_ASSIGNMENT -        ASS_PARTNER              - ASS_POLE

Entity2 Belgique - Assistance (OPS) - Médical        - Nan                             - MEDICAL

Entity1 France - Assistance         - Crises         - Crises                          - CRISES

Entity1 France - Commerciale        - RTC            - Relation Téléphonique Clientèle - CLIENTS
'''