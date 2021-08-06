# !/usr/bin/python
# coding=utf-8

from sklearn import svm
import preprocessing
from pandas.core.frame import DataFrame


def svm_train_and_test(x_train, x_test, y_train, y_test):
    #print(x_train, x_test, y_train, y_test)
    x = x_train.values.tolist()
    x = [i[2] for i in x]
    y = y_train.values.tolist()
    x2 = x_test.values.tolist()
    x2_id = [i[0] for i in x2]
    x2 = [i[2] for i in x2]
    y2 = y_test.values.tolist()
    clf = svm.SVC(gamma='scale')
    clf.fit(x, y)
    # clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
    score = clf.score(x2, y2)
    print(x2_id)
    print(clf.predict(x2))
    print(y2)
    return score


def svm_run(train_data_frame):
    x = train_data_frame.values.tolist()
    for i in range(len(x)):  # 矩阵展开成向量，测试 16900维
        print(i)
        x_i_2_temp = []
        for j in range(len(x[i][2])):
            x_i_2_temp += x[i][2][j]
        x[i][2] = x_i_2_temp
    train_data_frame = DataFrame(x, columns=['id', 'category', 'word_vector', 'label'])
    x_train, x_test, y_train, y_test = preprocessing.split_train_and_dev(train_data_frame)
    score = svm_train_and_test(x_train, x_test, y_train, y_test)
    print(score)

