import csv as csv
import numpy as numpy


def testing():
    prediction_file = open('../output/gender.csv', 'w', newline='')
    prediction_object = csv.writer(prediction_file)
    test_file = open('../data/test.csv', newline='')
    test_object = csv.reader(test_file)
    header = next(test_object)

    prediction_object.writerow(["PassengerId", "Survived"])
    for person in test_object:
        if person[3] == "female":
            prediction_object.writerow([person[0], 1])

        else:
            prediction_object.writerow([person[0], 0])

    prediction_file.close()
    test_file.close()

testing()
