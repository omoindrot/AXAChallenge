import pandas as pd
import numpy as np
from datetime import datetime

# Here we create a simple submission, with the mean taken on all data
# Variants: different mean with day of the week, entreprise, time_slot

X = pd.read_csv('data/train_2011_2012.csv', sep=';')
Y = np.asarray(X)

res = pd.read_csv('submission.txt', sep='\t')
Y_res = np.asarray(res)

# set of type of companies 'ASS_ASSIGNMENT'
S = set(Y[:, 12])

# set of type of companies in the test set
S_test = set(Y_res[:, 1])

# days, from monday to sunday
days = set(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])

days_from_int = {0: "Lundi",
                 1: "Mardi",
                 2: "Mercredi",
                 3: "Jeudi",
                 4: "Vendredi",
                 5: "Samedi",
                 6: "Dimanche"}

# hours, from 0 to 23
time_slots = set(Y[:, 6])
#time_slots = set(Y_res[:, 0].split(' ')[1])

mean_call = {}

for company in S_test:
    # We get all the ids of elements of the company
    idx = np.where(Y[:, 12] == company)
    Y_company = Y[idx]
    print company
    #
    mean_call[company] = {}
    for day in days:
        # We get all the ids of elements on a day
        idx2 = np.where(Y_company[:, 4] == day)
        Y_company_day = Y_company[idx2]
        mean_call[company][day] = {}
        for time_slot in time_slots:
            # We get all the ids of elements at a precise time slot
            idx3 = np.where(Y_company_day[:, 6] == time_slot)
            Y_company_day_time = Y_company[idx3]
            mean_call[company][day][time_slot] = np.mean(Y_company_day_time[:, 83])

f1 = open('submission.txt', 'r')
submission = f1.readlines()
f1.close()

f2 = open('submission/mean_company_day_time.txt', 'w')
f2.write(submission[0])

for i in range(1, len(submission)):
    temp = submission[i].split('\r')[0]
    temp = temp.split('\t')
    # date
    date = temp[0]
    temp1 = date.split(' ')
    # year
    year = int(temp1[0].split('-')[0])
    # month
    month = int(temp1[0].split('-')[1])
    # day
    day = int(temp1[0].split('-')[2])
    # hour
    hour = temp1[1].split(':')[0]
    hour = int(hour)
    # company
    company = temp[1]
    day_of_week = days_from_int[datetime(year, month, day).weekday()]
    # put the result
    temp[2] = '%f' % (mean_call[company][day_of_week][hour])
    res = temp[0]+'\t'+temp[1]+'\t'+temp[2]+'\r\n'
    f2.write(res)

f2.close()

# Score (MSE) for mean of data / company / day / time: 724.40
# Score (MSE) for mean of data / company:  682.88
# Score (MSE) for all 0: 819.61






# Example of element in X
print 'Example of a line in X (with largest phone calls number)'
print X.iloc[5391146].to_string()

