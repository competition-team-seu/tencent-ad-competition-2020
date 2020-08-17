#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def generate_age_gender_dis(all_pred_input):
    gender1 = np.sum(all_pred_input[:, :10], axis=1)
    gender1 = gender1[:, np.newaxis]

    gender2 = np.sum(all_pred_input[:, 10:], axis=1)
    gender2 = gender2[:, np.newaxis]

    gender_pred_5_input = np.concatenate([gender1, gender2], axis=1)

    age1 = np.sum(all_pred_input[:, [0, 10]], axis=1)
    age1 = age1[:, np.newaxis]

    age2 = np.sum(all_pred_input[:, [1, 11]], axis=1)
    age2 = age2[:, np.newaxis]

    age3 = np.sum(all_pred_input[:, [2, 12]], axis=1)
    age3 = age3[:, np.newaxis]

    age4 = np.sum(all_pred_input[:, [3, 13]], axis=1)
    age4 = age4[:, np.newaxis]

    age5 = np.sum(all_pred_input[:, [4, 14]], axis=1)
    age5 = age5[:, np.newaxis]

    age6 = np.sum(all_pred_input[:, [5, 15]], axis=1)
    age6 = age6[:, np.newaxis]

    age7 = np.sum(all_pred_input[:, [6, 16]], axis=1)
    age7 = age7[:, np.newaxis]

    age8 = np.sum(all_pred_input[:, [7, 17]], axis=1)
    age8 = age8[:, np.newaxis]

    age9 = np.sum(all_pred_input[:, [8, 18]], axis=1)
    age9 = age9[:, np.newaxis]

    age10 = np.sum(all_pred_input[:, [9, 19]], axis=1)
    age10 = age10[:, np.newaxis]

    age_pred_input = np.concatenate([age1, age2, age3, age4, age5, age6, age7, age8, age9, age10], axis=1)

    return gender_pred_5_input, age_pred_input


num = 0
path = "./submission_temp/"  # 文件夹目录
files = os.listdir(path)  # 得到文件夹下的所有文件名称
gender_test = list()
age_test = list()
for file in files:  # 遍历文件夹
    if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
        lstm_result = np.load((path + "/" + file))
        print(file)
        if lstm_result.shape[0] > 1000000:
            lstm_result = lstm_result[900000:]
            print(lstm_result.shape)
        if lstm_result.shape[1] == 20:
            num = num + 1
            print(lstm_result.shape)
            gender_lstm_result, age_lstm_result = generate_age_gender_dis(lstm_result)
            print(gender_lstm_result.shape)
            gender_test.append(gender_lstm_result)
            print(age_lstm_result.shape)
            age_test.append(age_lstm_result)
        elif lstm_result.shape[1] == 10:
            num = num + 1
            print(lstm_result.shape)
            age_test.append(lstm_result)
        else:
            num = num + 1
            print(lstm_result.shape)
            gender_test.append(lstm_result)

print()
print("The num is " + str(num))
print(np.shape(gender_test))
print(np.shape(age_test))
print()

gender_factor = np.ones(np.shape(gender_test))
age_factor = np.ones(np.shape(age_test))

print("gender")
print("===" * 30)

gender_result_prob = np.zeros([1, 2])
for item in range(len(gender_test)):
    gender_result_prob = gender_result_prob + gender_test[item] * gender_factor[item]

print("final result")
gender_average_result = np.argmax(gender_result_prob, axis=1) + 1

print("age")
print("===" * 30)
age_result_prob = np.zeros([1, 10])
for item in range(len(age_test)):
    age_result_prob = age_result_prob + age_test[item] * age_factor[item]

print("final result")
age_average_result = np.argmax(age_result_prob, axis=1) + 1

data_list = pd.read_pickle('all_log_agg_wide_semi_input_1.pkl')

submission = pd.DataFrame(columns=['user_id', 'predicted_age', 'predicted_gender'])
print(submission.size)

submission['user_id'] = data_list[3000000:]["user_id"]

submission['predicted_age'] = age_average_result

submission['predicted_gender'] = gender_average_result

submission.to_csv('submission_result/submission0722_ensemble_9.csv', index=False)

load_labels = pd.read_csv('submission_result/submission0722_ensemble_1.csv')
print(accuracy_score(submission['predicted_gender'], load_labels['predicted_gender']))
print(accuracy_score(submission['predicted_age'], load_labels['predicted_age']))
