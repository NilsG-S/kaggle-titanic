import csv as csv
import numpy as numpy


def training():
    train_file = open('../data/train.csv', newline='')
    train_object = csv.reader(train_file)
    next(train_object)

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

    print(survive_table)

    train_file.close()

training()
