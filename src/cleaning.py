import pandas as pandas
from sklearn import tree


def tree_fill(data, target_name, feature_names):
    nan_free = data[target_name + feature_names].dropna()

    target = nan_free[target_name].values
    features = nan_free[feature_names].values

    train_tree = tree.DecisionTreeClassifier()
    train_tree.fit(features, target)

    nan_features = data[data[target_name].isnull()][feature_names].values

    nan_fill = train_tree.predict(nan_features)

    data = data[data[target_name].map(
        lambda x: nan_fill.pop(0) if pandas.isnull(x) else True
    )]


def age_gathering(ages, row):
    # Append current row value to correct entry in ages
    ages[(row['Sex'], row['Pclass'])].append(row['Age'])


def cleaning(age_medians, ports, row):
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

    if not pandas.isnull(row['Embarked']):
        row['Port'] = ports[row['Embarked']]

    return row


def clean(input_path, output_path):
    # Open the training data set
    data = pandas.read_csv(input_path, header=0)

    # Data cleaning

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

    ports = {
        'C': 0,
        'Q': 1,
        'S': 2
    }

    # Detect rows with NaN values
    # data[data.isnull().values]

    data = data.apply(lambda x: cleaning(age_medians, ports, x), axis=1)

    data.to_csv(output_path, index=False)
