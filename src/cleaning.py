import pandas as pandas
from sklearn import tree


# Detect rows with NaN values
# data[data.isnull().values]

# Converting

def converting(row):
    if row['Sex'] == 'female':
        row['Gender'] = 0
    elif row['Sex'] == 'male':
        row['Gender'] = 1

    if row["Embarked"] == "C":
        row['Port'] = 0
    elif row["Embarked"] == "Q":
        row['Port'] = 1
    elif row["Embarked"] == "S":
        row['Port'] = 2

    # Fair bracket 1 is for fare =< 10
    if row["Fare"] <= 10:
        row["FareGroup"] = 1
    # Fair bracket 2 is for fare =< 20
    elif row["Fare"] <= 20:
        row["FareGroup"] = 2
    # Fair bracket 3 is for fare =< 30
    elif row["Fare"] <= 30:
        row["FareGroup"] = 3
    # Fair bracket 4 is for fare > 30
    elif row["Fare"] > 30:
        row["FareGroup"] = 4

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

def cleaning(age_medians, fare_fill, row):
    if pandas.isnull(row['Age']):
        row['AgeFill'] = age_medians[(row['Sex'], row['Pclass'])]
        row['AgeIsNull'] = 1
    else:
        row['AgeFill'] = row['Age']
        row['AgeIsNull'] = 0

    if pandas.isnull(row["FareGroup"]):
        row["FareGroup"] = fare_fill.pop(0)
        row["FareIsNull"] = 1
    else:
        row["FareIsNull"] = 0

    return row


def clean(input_path, output_path):
    # Open the training data set
    data = pandas.read_csv(input_path, header=0)

    # Converting

    data = data.apply(lambda x: converting(x), axis=1)

    # Gathering

    ages = {}
    age_medians = {}

    for sex in ("female", "male"):
        for p_class in range(1, 4):
            ages[(sex, p_class)] = []
            age_medians[(sex, p_class)] = 0

    data.apply(lambda x: age_gathering(ages, x), axis=1)

    for sex in ("female", "male"):
        for p_class in range(1, 4):
            age_medians[(sex, p_class)] = \
                pandas.Series(ages[(sex, p_class)]).median()

    fare_fill = tree_fill(data, ["FareGroup"], ["Pclass", "Gender"])

    # Cleaning

    data = data.apply(lambda x: cleaning(age_medians, fare_fill, x), axis=1)

    # Output

    data.to_csv(output_path, index=False)
