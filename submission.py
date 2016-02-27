# encoding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from preprocessing import lstm_test_set

# Submission


def create_submission(X_cleaned, model, companies_set, input_days=4, flat=False):
    test_days = [date(2012, 1, 3), date(2012, 2, 8), date(2012, 3, 12), date(2012, 4, 16),
                 date(2012, 5, 19), date(2012, 6, 18), date(2012, 7, 22), date(2012, 8, 21),
                 date(2012, 9, 20), date(2012, 10, 24), date(2012, 11, 26), date(2012, 12, 28)]

    X_test_companies = lstm_test_set(X_cleaned, companies_set, test_days, input_days=input_days, flat=flat)

    y_predicted_companies = {}
    for company in companies_set:
        y_predicted_companies[company] = model.predict({'input': X_test_companies[company]})['output']
        y_predicted_companies[company][y_predicted_companies[company]<0] = 0.

    # y_predicted_companies[company] of shape 12, 48

    f1 = open('submission.txt', 'r')
    submission = f1.readlines()
    f1.close()

    f2 = open('submission/lstm_%ddays_64hdim.txt' % input_days, 'w')
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
