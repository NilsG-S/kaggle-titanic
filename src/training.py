import csv as csv
import numpy as numpy


def training():
    train_file = open('../data/train.csv', newline='')
    train_file_object = csv.reader(train_file)
    header = next(train_file_object)

    data = []
    for row in train_file_object:
        data.append(row)

    data = numpy.array(data)

    num_passengers = 0
    women = 0
    women_survived = 0
    men = 0
    men_survived = 0
    for person in data:
        num_passengers += 1

        if person[4] == "female":
            women += 1

            if person[1].astype(numpy.float) == 1:
                women_survived += 1

        else:
            men += 1

            if person[1].astype(numpy.float) == 1:
                men_survived += 1

    prop_women_survived = women_survived / women
    prop_men_survived = men_survived / men

    print(prop_women_survived)
    print(prop_men_survived)

    train_file.close()

training()
