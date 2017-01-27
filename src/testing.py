import csv as csv
import numpy as numpy


def testing():
    prediction_file = open('../output/genderclassfare.csv', 'w', newline='')
    prediction_object = csv.writer(prediction_file)
    test_file = open('../data/test.csv', newline='')
    test_object = csv.reader(test_file)
    next(test_object)
    model_file = open('../model/model.csv', newline='')
    model_object = csv.reader(model_file)
    next(model_object)

    survived = {}

    for row in model_object:
        survived[(row[0], int(row[1]), int(row[2]))] = row[3]

    fare_groups = {
        0: 1,
        1: 2,
        2: 3,
        3: 4
    }

    bin_width = 10

    prediction_object.writerow(["PassengerId", "Survived"])
    for row in test_object:
        if not row[8]:
            fare = 0.0
        else:
            fare = float(row[8])

        fare_group = fare_groups.get(fare // bin_width, 4)
        prediction_object.writerow(
            [row[0], survived[(row[3], int(row[1]), fare_group)]]
        )

    prediction_file.close()
    test_file.close()
    model_file.close()

testing()
