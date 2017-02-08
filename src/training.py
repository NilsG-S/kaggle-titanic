import csv as csv
import pandas as pandas


def data_handler(pas, sur, ages, fares, width, row):
    # Performs a floor operation at intervals of bin_width
    # If the result cannot be found, return fare bracket 4
    fare = fares.get(row[9] // width, 4)
    # Add 1 to the given category in the passengers dict
    pas[(row[4], row[2], fare)] += 1
    # Append current row value to correct entry in ages
    ages[(row[4], row[2], fare)].append(row[5])
    if int(row[1]) == 1:
        # If the passenger survived add 1 to the correct category
        sur[(row[4], row[2], fare)] += 1


def training():
    # Open the file to hold the model (trained data)
    model_file = open('../model/model.csv', 'w', newline='')
    model_object = csv.writer(model_file)
    # Open the training data set
    train_file = open('../data/train.csv', newline='')
    train_object = csv.reader(train_file)
    # Skip the header row
    next(train_object)

    data = pandas.read_csv('../data/train.csv', header=0)

    # Dictionaries with keys based on sex, passenger class, and fare bracket
    passengers = {}
    survived = {}
    ages = {}

    # Initializing the dictionaries to 0
    for sex in ("female", "male"):
        for p_class in range(1, 4):
            for fare in range(1, 5):
                passengers[(sex, p_class, fare)] = 0
                survived[(sex, p_class, fare)] = 0
                ages[(sex, p_class, fare)] = []

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

    data.apply(lambda x: data_handler(passengers, survived, ages, fare_group,
                                      bin_width, x), axis=1)

    # data['Gender'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # median_ages[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna().median()

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
    train_file.close()

training()
