import pandas as pandas


def generate(row):
    row['FamilySize'] = row['SibSp'] + row['Parch']
    row['Age*Class'] = row["Age"] * row["Pclass"]

    return row


def gen_features(input_path, output_path):
    # Open the cleaned training data set
    data = pandas.read_csv(input_path, header=0)

    # Feature engineering

    data = data.apply(lambda x: generate(x), axis=1)
    # data = data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

    data.to_csv(output_path, index=False)
