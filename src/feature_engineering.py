import pandas as pandas


def generate(row):
    row['FamilySize'] = row['SibSp'] + row['Parch']
    row['Age*Class'] = row["AgeFill"] * row["Pclass"]

    return row


def gen_features():
    # Open the cleaned training data set
    data = pandas.read_csv('../cleaned/clean_train.csv', header=0)

    # Feature engineering

    data = data.apply(lambda x: generate(x), axis=1)

    data.to_csv('../engineered/engineer_train.csv', index=False)

gen_features()
