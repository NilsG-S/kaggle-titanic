import pandas as pandas


def model_gathering(survived, row):
    survived[(row['Gender'], int(row['Pclass']), int(row['Fare']))] = \
        row['Prediction']


def results(sur, fares, width, row):
    if not row['Fare']:
        fare = 0.0
    else:
        fare = row['Fare']

    fare_group = fares.get(fare // width, 4)

    return [row['PassengerId'],
            sur[(row['Gender'],
                 row['Pclass'],
                 fare_group)]]


def test():
    test_data = pandas.read_csv('../cleaned/clean_test.csv', header=0)
    model_data = pandas.read_csv('../model/model.csv', header=0)

    survived = {}

    model_data.apply(lambda x: model_gathering(survived, x), axis=1)

    fare_groups = {
        0: 1,
        1: 2,
        2: 3,
        3: 4
    }

    bin_width = 10

    output = (
        test_data.apply(
            lambda x: results(survived, fare_groups, bin_width, x),
            axis=1
        ).tolist()
    )
    output.insert(0, ["PassengerId", "Survived"])

    pandas.DataFrame(output).to_csv('../output/genderclassfare.csv', index=False, header=0)
