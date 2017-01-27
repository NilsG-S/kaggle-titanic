import csv as csv
import numpy as numpy


def testing():
    prediction_file = open('../output/genderclassfare.csv', 'w', newline='')
    prediction_object = csv.writer(prediction_file)
    test_file = open('../data/test.csv', newline='')
    test_object = csv.reader(test_file)
    next(test_object)
    train_file = open('../data/train.csv', newline='')
    train_object = csv.reader(train_file)
    next(train_object)

    # Training

    data = []
    for row in train_object:
        data.append(row)
    data = numpy.array(data)

    fare_ceiling = 40
    data[data[0::, 9].astype(numpy.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

    fare_bracket_size = 10
    number_of_price_brackets = fare_ceiling // fare_bracket_size

    number_of_classes = len(numpy.unique(data[0::, 2]))

    survive_table = numpy.zeros([2, number_of_classes, number_of_price_brackets], float)

    for i in range(number_of_classes):
        for j in range(number_of_price_brackets):
            women_only_stats = data[
                (data[0::, 4] == "female")
                & (data[0::, 2].astype(numpy.float)
                   == i + 1)
                & (data[0::, 9].astype(numpy.float)
                   >= j * fare_bracket_size)
                & (data[0::, 9].astype(numpy.float)
                   < (j + 1) * fare_bracket_size),
                1
            ]

            men_only_stats = data[
                (data[0::, 4] != "female")
                & (data[0::, 2].astype(numpy.float)
                   == i + 1)
                & (data[0::, 9].astype(numpy.float)
                   >= j * fare_bracket_size)
                & (data[0::, 9].astype(numpy.float)
                   < (j + 1) * fare_bracket_size),
                1
            ]

            survive_table[0, i, j] = \
                numpy.mean(women_only_stats.astype(numpy.float))
            survive_table[1, i, j] = \
                numpy.mean(men_only_stats.astype(numpy.float))

    survive_table[survive_table != survive_table] = 0

    survive_table[survive_table < 0.5] = 0
    survive_table[survive_table >= 0.5] = 1

    # Testing

    prediction_object.writerow(["PassengerId", "Survived"])
    for person in test_object:

        for j in range(4):

            try:
                person[8] = float(person[8])

            except:
                bin_fare = 3 - float(person[1])
                break

            if person[8] > 40:

                bin_fare = 3
                break

            if person[8] >= j * 10 and person[8] < (j + 1) * 10:
                bin_fare = j
                break

        if person[3] == "female":
            prediction_object.writerow([person[0], "%d" % int(survive_table[0, int(person[1]) - 1, int(bin_fare)])])

        else:
            prediction_object.writerow([person[0], "%d" % int(survive_table[1, int(person[1]) - 1, int(bin_fare)])])

    prediction_file.close()
    test_file.close()
    train_file.close()

testing()
