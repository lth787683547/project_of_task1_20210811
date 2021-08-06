# !/usr/bin/python
# coding=utf-8
import numpy
import pandas
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, KFold
from pandas.core.frame import DataFrame


def read_file(file_name, train_or_test, padding, max_length):
    """从tsv文件内读取数据"""
    zero_list_dim_100 = []
    for _ in range(100):
        zero_list_dim_100.append(0.0)
    #zero_series_dim_100 = pd.array(data=zero_vec_dim_100,dtype=numpy.float64)
    if train_or_test == 'train':
        pkl_file_name = file_name[:-3] + 'pkl'
        if os.path.exists(pkl_file_name):
            #with open(pkl_file_name, 'rb') as f1:
            #    train_data_frame = pickle.load(f1)
            train_data_frame = pd.read_pickle(pkl_file_name)
            return train_data_frame
        else:
            train_data_frame = pd.read_csv(file_name,sep='\t',header=0, names=['id','category','word_vector','label'])
            x = train_data_frame.values.tolist()
            for i in range(len(x)):
                x[i][2] = (eval(x[i][2]))
                if padding:
                    len_this = len(x[i][2])
                    for _ in range(max_length-len_this):
                        x[i][2].append(zero_list_dim_100)
            train_data_frame = DataFrame(x, columns=['id', 'category', 'word_vector', 'label'])
            #with open(file_name[:-3]+'pkl', "wb") as f:
            #    pickle.dump(train_data_frame, f)
            train_data_frame.to_pickle(file_name[:-3] + 'pkl')
            # train_data_frame.to_csv(file_name[:-3]+'csv')
            return train_data_frame
    elif train_or_test == 'test':
        pkl_file_name = file_name[:-3] + 'pkl'
        csv_file_name = file_name[:-3] + 'csv'
        if os.path.exists(pkl_file_name):
            #with open(pkl_file_name, 'rb') as f1:
            test_data_frame = pd.read_pickle(pkl_file_name)
            return test_data_frame
        #if os.path.exists(csv_file_name):
        #    test_data_frame = pd.read_csv(csv_file_name)
        #    return test_data_frame

        else:
            test_data_frame = pd.read_csv(file_name,sep='\t',header=0, names=['id','category','word_vector'])
            x = test_data_frame.values.tolist()
            for i in range(len(x)):
                x[i][2] = (eval(x[i][2]))
                if padding:
                    len_this = len(x[i][2])
                    for _ in range(max_length-len_this):
                        x[i][2].append(zero_list_dim_100)
            test_data_frame = DataFrame(x, columns=['id', 'category', 'word_vector'])
            #with open(file_name[:-3]+'pkl', "wb") as f:
            #    pickle.dump(test_data_frame, f)
            test_data_frame.to_pickle(file_name[:-3]+'pkl')
            #test_data_frame.to_csv(file_name[:-3] + 'csv')
            return test_data_frame
    else:
        raise Exception("invalid augument train_or_test, please use 'train' or 'test'")


def split_train_and_dev(data_frame_all, test_size=0.1):
    """
    由于没有测试集答案，将训练集分成训练集和测试集，以评估效果，比例默认为10% 90%
    :param data_frame_all: 所有的训练集原始数据
    :return: 训练集[id, category, word_vector],训练集[label],后两个测试集对应
    """
    x = data_frame_all[['id','category','word_vector']]
    y = data_frame_all['label']
    # print(x)
    # print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)
    # 先设置随机数种子为1 评估效果
    return x_train, x_test, y_train, y_test


def split_train_and_dev_k_fold(data_frame_all):
    """
    k折交叉验证
    :param data_frame_all: 所有训练数据
    :return: [[第一批x_train, x_test, y_train, y_test],[],...,[第n批x_train, x_test, y_train, y_test]]
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    x = data_frame_all[['id', 'category', 'word_vector']]
    y = data_frame_all['label']
    print(x.shape, y.shape)
    result_list = []
    X = np.arange(x.shape[0])
    for train_index, test_index in kf.split(X):
        x_train, x_test, y_train, y_test = x.iloc[train_index], x.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        result_list.append([x_train, x_test, y_train, y_test])
    return result_list


def get_statistics(data_frame_all):
    length = data_frame_all.shape[0]
    word_vector_list = data_frame_all['word_vector'].values.tolist()
    print("length:", length)
    #print("word_vector_list:", word_vector_list)
    length_list = []
    for i in range(len(word_vector_list)):
        length_list.append(len(word_vector_list[i]))

    length_list.sort()
    min = length_list[0]
    max = length_list[len(length_list)-1]
    print('min:', min)
    print('max:', max)
    sum = 0
    for i in length_list:
        sum += i
    average = sum/len(length_list)
    print('average:', average)
    middle = length_list[int(len(length_list)/2)]
    print('middle:', middle)

    label_list = data_frame_all['label'].values.tolist()
    benign = 0
    malicious = 0
    for i in label_list:
        if i == 2:
            benign += 1
        elif i == 1:
            malicious += 1
        else:
            print('error')
    print('begign:', benign)
    print('malicious:', malicious)
    return min, max, average, middle, benign, malicious


def get_statistics_test(data_frame_all):
    length = data_frame_all.shape[0]
    word_vector_list = data_frame_all['word_vector'].values.tolist()
    print("length:", length)
    #print("word_vector_list:", word_vector_list)
    length_list = []
    for i in range(len(word_vector_list)):
        length_list.append(len(word_vector_list[i]))

    length_list.sort()
    min = length_list[0]
    max = length_list[len(length_list)-1]
    print('min:', min)
    print('max:', max)
    sum = 0
    for i in length_list:
        sum += i
    average = sum/len(length_list)
    print('average:', average)
    middle = length_list[int(len(length_list)/2)]
    print('middle:', middle)
    return min, max, average, middle


if __name__ == '__main__':
    train_file_name = "F:\\study\\postgraduate\\CDMC2021\\Task1\\Training.tsv"
    test_file_name = "F:\\study\\postgraduate\\CDMC2021\\Task1\\Testing.tsv"
    test_data_frame = read_file(test_file_name, 'test', padding=True, max_length=169)
    train_data_frame = read_file(train_file_name, 'train', padding=True, max_length=169)
    # x_train, x_test, y_train, y_test = split_train_and_dev_k_fold(train_data_frame)
    # print(x_train, x_test, y_train, y_test)
    min, max, average, middle, benign, malicious = get_statistics(train_data_frame)
    min, max, average, middle = get_statistics_test(test_data_frame)