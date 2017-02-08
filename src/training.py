import csv as csv
import pandas as pandas


def age_data(ages, row):
    # Append current row value to correct entry in ages
    ages[(row['Sex'], row['Pclass'])].append(row['Age'])


def data_cleaning(age_medians, row):
    if row['Sex'] == 'female':
        row['Gender'] = 0
    else:
        row['Gender'] = 1

    if pandas.isnull(row['Age']):
        row['AgeFill'] = age_medians[(row['Sex'], row['Pclass'])]
        row['AgeIsNull'] = 1
    else:
        row['AgeFill'] = row['Age']
        row['AgeIsNull'] = 0

    return row


def data_gathering(pas, sur, fares, width, row):
    # Performs a floor operation at intervals of bin_width
    # If the result cannot be found, return fare bracket 4
    fare = fares.get(row[9] // width, 4)
    # Add 1 to the given category in the passengers dict
    pas[(row[4], row[2], fare)] += 1
    if int(row[1]) == 1:
        # If the passenger survived add 1 to the correct category
        sur[(row[4], row[2], fare)] += 1


def training():
    # Open the file to hold the model (trained data)
    model_file = open('../model/model.csv', 'w', newline='')
    model_object = csv.writer(model_file)
    # Open the training data set
    data = pandas.read_csv('../data/train.csv', header=0)

    # Data cleaning

    ages = {}
    age_medians = {}

    for sex in ("female", "male"):
        for p_class in range(1, 4):
            ages[(sex, p_class)] = []
            age_medians[(sex, p_class)] = 0

    data.apply(lambda x: age_data(ages, x), axis=1)

    for sex in ("female", "male"):
        for p_class in range(1, 4):
            age_medians[(sex, p_class)] = \
                pandas.Series(ages[(sex, p_class)]).median()

    data = data.apply(lambda x: data_cleaning(age_medians, x), axis=1)

    # Data gathering

    # Dictionaries with keys based on sex, passenger class, and fare bracket
    passengers = {}
    survived = {}

    # Initializing the dictionaries to 0
    for sex in ("female", "male"):
        for p_class in range(1, 4):
            for fare in range(1, 5):
                passengers[(sex, p_class, fare)] = 0
                survived[(sex, p_class, fare)] = 0

    # A dictionary to match an integer value to a fare bracket
    # Fair bracket 1 is for fare =< 10
    # Fair bracket 2 is for fare =< 20
    # Fair bracket 3 is for fare =< 30
    # Fair bracket 4 is for fare > 30
    fare_group = {
        0: 1,
        1: 2,
        2: 3,
        3: 4
    }

    # Width of the fare brackets
    bin_width = 10

    data.apply(lambda x: data_gathering(passengers, survived, fare_group,
                                        bin_width, x), axis=1)

    # Data analysis

    model_object.writerow(["Sex", "Pclass", "Fare", "Prediction"])
    for sex in ("female", "male"):
        for p_class in range(1, 4):
            for fare in range(1, 5):
                if passengers[(sex, p_class, fare)] == 0:
                    # If there were no passengers, set the survival rate to 0
                    rate = 0.0
                else:
                    # Otherwise, the survival rate is survived / passengers
                    rate = survived[(sex, p_class, fare)] \
                           / passengers[(sex, p_class, fare)]

                if rate < 0.5:
                    # If the rate is less than 0.5, assume nobody survived
                    prediction = 0
                else:
                    # Otherwise, assume everyone survived
                    prediction = 1

                # Write the prediction to the model file
                model_object.writerow([sex, p_class, fare, prediction])

    model_file.close()

training()
