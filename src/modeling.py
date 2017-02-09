import pandas as pandas


def data_gathering(pas, sur, fares, width, row):
    # Performs a floor operation at intervals of bin_width
    # If the result cannot be found, return fare bracket 4
    fare = fares.get(row['Fare'] // width, 4)
    # Add 1 to the given category in the passengers dict
    pas[(row['Gender'], row['Pclass'], fare)] += 1
    if int(row['Survived']) == 1:
        # If the passenger survived add 1 to the correct category
        sur[(row['Gender'], row['Pclass'], fare)] += 1


def gen_model():
    # Open the training data set
    data = pandas.read_csv('../cleaned/clean_train.csv', header=0)

    # Data gathering

    # Dictionaries with keys based on gender, passenger class, and fare bracket
    passengers = {}
    survived = {}

    # Initializing the dictionaries to 0
    for gender in (0, 1):
        for p_class in range(1, 4):
            for fare in range(1, 5):
                passengers[(gender, p_class, fare)] = 0
                survived[(gender, p_class, fare)] = 0

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

    rows = [["Gender", "Pclass", "Fare", "Prediction"]]
    for gender in (0, 1):
        for p_class in range(1, 4):
            for fare in range(1, 5):
                if passengers[(gender, p_class, fare)] == 0:
                    # If there were no passengers, set the survival rate to 0
                    rate = 0.0
                else:
                    # Otherwise, the survival rate is survived / passengers
                    rate = survived[(gender, p_class, fare)] \
                           / passengers[(gender, p_class, fare)]

                if rate < 0.5:
                    # If the rate is less than 0.5, assume nobody survived
                    prediction = 0
                else:
                    # Otherwise, assume everyone survived
                    prediction = 1

                # Write the prediction to the model file
                rows.append([gender, p_class, fare, prediction])

    pandas.DataFrame(rows).to_csv('../model/model.csv', index=False, header=0)
