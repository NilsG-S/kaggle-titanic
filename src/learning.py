import pandas as pandas
import numpy as numpy


def learn():
    engineer_data = pandas.read_csv('../engineered/engineer_train.csv', header=0)

    train_data = engineer_data.values

    print(train_data[0::, 5])

learn()
