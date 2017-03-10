import pandas as pandas


def generate(row):
    if row['SibSp'] + row['Parch'] == 1:
        row["IsAlone"] = 1
    else:
        row["IsAlone"] = 0

    row['Age*Class'] = row["Age"] * row["Pclass"]

    return row


def gen_features(input_path, output_path):
    # Open the cleaned training data set
    data = pandas.read_csv(input_path, header=0)

    # Feature engineering

    data = data.apply(lambda x: generate(x), axis=1)
    data = data.drop(['Parch', 'SibSp', 'Ticket', 'Cabin'], axis=1)

    data.to_csv(output_path, index=False)
