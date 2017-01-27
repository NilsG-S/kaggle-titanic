import csv as csv
import numpy as numpy


def training():
    model_file = open('../model/model.csv', 'w', newline='')
    model_object = csv.writer(model_file)
    train_file = open('../data/train.csv', newline='')
    train_object = csv.reader(train_file)
    next(train_object)

    # New

    passengers = {}
    survived = {}

    for sex in ("female", "male"):
        for p_class in range(1, 4):
            for fare in range(1, 5):
                passengers[(sex, p_class, fare)] = 0
                survived[(sex, p_class, fare)] = 0

    fare_group = {
        0: 1,
        1: 2,
        2: 3,
        3: 4
    }

    bin_width = 10

    for row in train_object:
        fare = fare_group.get(float(row[9]) // bin_width, 4)
        passengers[(row[4], int(row[2]), fare)] += 1
        if int(row[1]) == 1:
            survived[(row[4], int(row[2]), fare)] += 1

    model_object.writerow(["Sex", "Pclass", "Fare", "Prediction"])
    for sex in ("female", "male"):
        for p_class in range(1, 4):
            for fare in range(1, 5):
                if passengers[(sex, p_class, fare)] == 0:
                    rate = 0.0
                else:
                    rate = survived[(sex, p_class, fare)] \
                           / passengers[(sex, p_class, fare)]

                if rate < 0.5:
                    prediction = 0
                else:
                    prediction = 1

                model_object.writerow([sex, p_class, fare, prediction])

    model_file.close()
    train_file.close()

training()
