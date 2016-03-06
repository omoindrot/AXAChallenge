# encoding: utf-8

from datetime import date

# Submission


def create_submission(name, y_pred, days_test):
    f1 = open('submission.txt', 'r')
    submission = f1.readlines()
    f1.close()

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

