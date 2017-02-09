import pandas as pandas


def age_gathering(ages, row):
    # Append current row value to correct entry in ages
    ages[(row['Sex'], row['Pclass'])].append(row['Age'])


def cleaning(age_medians, row):
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

    data = data.apply(lambda x: cleaning(age_medians, x), axis=1)

    data.to_csv(output_path, index=False)
