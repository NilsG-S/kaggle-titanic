import pandas as pandas
from sklearn import tree


def learn():
    engineer_train = pandas.read_csv(
        '../engineered/engineer_train.csv', header=0
    )
    engineer_test = pandas.read_csv(
        '../engineered/engineer_test.csv', header=0
    )

    train_features = engineer_train[
        ["AgeFill",
         "Fare",
         "Gender",
         "Pclass",
         "FamilySize"]
    ].values

    engineer_test = engineer_test[engineer_test["Fare"].map(
        lambda x: False if pandas.isnull(x) else True
    )]

    test_features = engineer_test[
        ["AgeFill",
         "Fare",
         "Gender",
         "Pclass",
         "FamilySize"]
    ].values

    target = engineer_train["Survived"].values

    train_tree = tree.DecisionTreeClassifier()
    train_tree.fit(train_features, target)

    print(train_tree.feature_importances_)
    print(train_tree.predict(test_features))

learn()
