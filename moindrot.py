# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date


# Here we create a simple submission, with the mean taken on all data
# Variants: different mean with day of the week, entreprise, time_slot

X = pd.read_csv('data/train_2011_2012.csv', sep=';')

res = pd.read_csv('submission.txt', sep='\t')


# We only keep three columns: DATE, ASS_ASSIGNMENT, CSPL_CALLS
X = pd.DataFrame({'DATE': X['DATE'], 'ASS_ASSIGNMENT': X['ASS_ASSIGNMENT'], 'CALLS': X['CSPL_CALLS']})

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

# TODO : try to create a column of datetime values instead. Use it somewhat? Check the doc
index = pd.date_range('2011-01-01 00:00:00', periods=2*365*24*2, freq='30MIN')

# We sort by date
X.sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR'], inplace=True)

# List of companies ('ASS_ASSIGNMENT')
X['ASS_ASSIGNMENT'].value_counts()
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

# Set of companies
companies_set = set(res['ASS_ASSIGNMENT'].value_counts().index)


# Select one company
company = 'SAP'
X_company = X[X['ASS_ASSIGNMENT'] == company]

# TODO: We have to cleanup the data.
# TODO: group together the 48 values of each day
# If multiple values for one date: sum them
# If no values for one date: put 0

# We should get here a clean DataFrame with every date available (except the dates from the test set)
index = pd.date_range('2011-01-01 00:00:00', periods=(2*365+1)*24*2, freq='30MIN')
index_days = pd.date_range('2011-01-01 00:00:00', periods=2*365+1, freq='D')
values = np.zeros((2*365+1, 48))  # 48 values for each day (48 time slots)
begin = date(2011, 1, 1)
for slot in index:
    i = (slot.date() - begin).days  # index in values to insert (0<=i<731)
    j = 2*slot.hour + slot.minute/30  # index in values to insert (0<=j<48)
    # Get all values in X_company for this time_slot
    values[i, j] += X_company[X_company['DATE'] == slot.isoformat(' ')+'.000']['CALLS'].sum()


X_train = pd.DataFrame(values, index_days)


# TODO: Create the training set


# TODO: Split training set and validation set
split_val = 0.2
X_train = X_train.shuffle()
X_val = X_train[:split_val*X_train.shape[0]]
X_train = X_train[split_val*X_train.shape[0]:]


# Test
Y = X[:100].copy()

