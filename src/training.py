import csv as csv
import numpy as numpy


def main():
    csv_file_object = csv.reader(open('../data/train.csv'))
    header = next(csv_file_object)

    data = []
    for row in csv_file_object:
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

main()
