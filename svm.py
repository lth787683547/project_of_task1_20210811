# !/usr/bin/python
# coding=utf-8

import gc

import numpy
from datetime import datetime
import preprocessing
from sklearn import svm
from pandas.core.frame import DataFrame
from monitor import get_current_memory_gb


def svm_train_and_test(x_train, x_test, y_train, y_test):
    #print(x_train, x_test, y_train, y_test)
    x = x_train.values.tolist()
    x = [i[2] for i in x]
    print("after deal x:",get_current_memory_gb())
    y = y_train.values.tolist()
    print("after deal y:", get_current_memory_gb())
    x2 = x_test.values.tolist()
    x2_id = [i[0] for i in x2]
    print("after deal x_test:", get_current_memory_gb())
    x2 = [i[2] for i in x2]
    y2 = y_test.values.tolist()
    print("after deal y_test:", get_current_memory_gb())
    clf = svm.SVC(gamma='scale')

    #del x_train, x_test, y_train, y_test
    #gc.collect()
    #print("after del:", get_current_memory_gb())
    print("train start", datetime.now())
    clf.fit(x, y)
    print("train finished", datetime.now())
    # clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    #print(x2_id)
    #print(clf.predict(x2))
    #print(y2)
    score = clf.score(x2, y2)
    return score


def svm_train_and_test_linear(x_train, x_test, y_train, y_test):
    #print(x_train, x_test, y_train, y_test)
    x = x_train.values.tolist()
    x = [i[2] for i in x]
    print("after deal x:", get_current_memory_gb())
    y = y_train.values.tolist()
    print("after deal y:", get_current_memory_gb())
    x2 = x_test.values.tolist()
    x2_id = [i[0] for i in x2]
    print("after deal x_test:", get_current_memory_gb())
    x2 = [i[2] for i in x2]
    y2 = y_test.values.tolist()
    print("after deal y_test:", get_current_memory_gb())
    clf = svm.LinearSVC(max_iter=250000)

    #del x_train, x_test, y_train, y_test
    #gc.collect()
    #print("after del:", get_current_memory_gb())
    start_time = datetime.now()
    print("train start", start_time)
    clf.fit(x, y)
    print("train finished:", datetime.now(), "\ttime:", datetime.now()-start_time)
    # clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    #print(x2_id)
    #print(clf.predict(x2))
    #print(y2)
    score = clf.score(x2, y2)
    return score


def svm_run(train_data_frame):
    # x = train_data_frame.head(20000).values.tolist()
    x = train_data_frame.values.tolist()
    for i in range(len(x)):  # 矩阵展开成向量，测试 16900维
        if i % 10000 == 0:
            print(i,get_current_memory_gb())
        #x[i][2] = numpy.array(x[i][2]).flatten().tolist() flatten占用额外空间
        #x[i][2] = numpy.array(x[i][2]).ravel().tolist()
        x_i_2_temp = []
        for j in range(len(x[i][2])):
            x_i_2_temp += x[i][2][j]
        x[i][2] = x_i_2_temp
        #del x_i_2_temp
        #gc.collect()
    train_data_frame = DataFrame(x, columns=['id', 'category', 'word_vector', 'label'])
    x_train, x_test, y_train, y_test = preprocessing.split_train_and_dev(train_data_frame)
    print("before train", get_current_memory_gb())
    score = svm_train_and_test(x_train, x_test, y_train, y_test)
    #score = svm_train_and_test_linear(x_train, x_test, y_train, y_test)
    print(score)

