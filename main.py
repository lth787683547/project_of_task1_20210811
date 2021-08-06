# !/usr/bin/python
# coding=utf-8

import preprocessing
import svm
from pandas.core.frame import DataFrame


if __name__ == '__main__':
    train_file_name = "F:\\study\\postgraduate\\CDMC2021\\Task1\\Training.tsv"
    test_file_name = "F:\\study\\postgraduate\\CDMC2021\\Task1\\Testing.tsv"
    train_data_frame = preprocessing.read_file(train_file_name, 'train', padding=False, max_length=169)
    svm.svm_run(train_data_frame)

