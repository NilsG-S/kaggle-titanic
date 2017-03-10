import math as math
import re as re

import pandas as pandas
from sklearn import tree


# Detect rows with NaN values
# data[data.isnull().values]

# Converting

def converting(row):
    if row['Sex'] == 'female':
        row['Sex'] = 0
    elif row['Sex'] == 'male':
        row['Sex'] = 1

    if row["Embarked"] == "C":
        row['Embarked'] = 0
    elif row["Embarked"] == "Q":
        row['Embarked'] = 1
    elif row["Embarked"] == "S":
        row['Embarked'] = 2

    # Fair bracket 1 is for fare =< 10
    if row["Fare"] <= 7.91:
        row["Fare"] = 1
    # Fair bracket 2 is for fare =< 20
    elif row["Fare"] <= 14.454:
        row["Fare"] = 2
    # Fair bracket 3 is for fare =< 30
    elif row["Fare"] <= 31:
        row["Fare"] = 3
    # Fair bracket 4 is for fare > 30
    elif row["Fare"] > 31:
        row["Fare"] = 4

    if not pandas.isnull(row['Age']):
        row["Age"] = math.ceil(row["Age"] / 16)

    title = re.search(' ([A-Za-z]+)\.', row["Name"]).group(1)

    if title == "Master":
        row["Title"] = 0
    elif title == "Mlle":
        row["Title"] = 1
    elif title == "Ms":
        row["Title"] = 1
    elif title == "Mr":
        row["Title"] = 2
    elif title == "Mme":
        row["Title"] = 3
    else:
        row["Title"] = 4

    return row


# Gathering

def age_gathering(ages, row):
    # Append current row value to correct entry in ages
    ages[(row['Sex'], row['Pclass'])].append(row['Age'])


def tree_fill(data, target_name, feature_names):
    nan_fill = []

    test = data[data[target_name].isnull().values]
    if test.empty:
        return nan_fill

    nan_free = data[target_name + feature_names].dropna()

    target = nan_free[target_name].values
    features = nan_free[feature_names].values

    train_tree = tree.DecisionTreeClassifier()
    train_tree.fit(features, target)

    nan_features = data[data[target_name].isnull().values][feature_names].values

    nan_fill = train_tree.predict(nan_features).tolist()

    return nan_fill


# Cleaning

def filling(col, fill, row):
    if pandas.isnull(row[col]):
        row[col] = fill.pop(0)

    return row


def cleaning(age_medians, row):
    if pandas.isnull(row['Age']):
        row['Age'] = age_medians[(row['Sex'], row['Pclass'])]

    return row


def clean(input_path, output_path):
    # Open the training data set
    data = pandas.read_csv(input_path, header=0)

    # Converting

    data = data.apply(lambda x: converting(x), axis=1)

    # Gathering

    ages = {}
    age_medians = {}

    for sex in (0, 1):
        for p_class in range(1, 4):
            ages[(sex, p_class)] = []
            age_medians[(sex, p_class)] = 0

    data.apply(lambda x: age_gathering(ages, x), axis=1)

    for sex in (0, 1):
        for p_class in range(1, 4):
            age_medians[(sex, p_class)] = \
                pandas.Series(ages[(sex, p_class)]).median()

    fare_fill = tree_fill(data, ["Fare"], ["Pclass", "Sex"])
    port_fill = tree_fill(data, ["Embarked"], ["Pclass", "Sex"])

    # Cleaning

    data = data.apply(
        lambda x: filling("Fare", fare_fill, x), axis=1
    )
    data = data.apply(
        lambda x: filling("Embarked", port_fill, x), axis=1
    )
    data = data.apply(lambda x: cleaning(age_medians, x), axis=1)

    # Output

    data.to_csv(output_path, index=False)
