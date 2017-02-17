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
         "FareGroup",
         "Gender",
         "Pclass",
         "FamilySize"]
    ].values

    test_features = engineer_test[
        ["AgeFill",
         "FareGroup",
         "Gender",
         "Pclass",
         "FamilySize"]
    ].values

    target = engineer_train["Survived"].values

    train_tree = tree.DecisionTreeClassifier()
    train_tree.fit(train_features, target)

    predictions = train_tree.predict(test_features)
    passengers = engineer_test["PassengerId"].values

    data = [["PassengerId", "Survived"]]
    for index in range(0, len(passengers)):
        data.append([passengers[index], predictions[index]])

    pandas.DataFrame(data).to_csv('../output/decision_tree.csv', index=False, header=0)
