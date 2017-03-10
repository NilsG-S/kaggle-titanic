import pandas as pandas
from sklearn.ensemble import RandomForestClassifier


def learn():
    engineer_train = pandas.read_csv(
        'engineered/engineer_train.csv', header=0
    )
    engineer_test = pandas.read_csv(
        'engineered/engineer_test.csv', header=0
    )

    train_features = engineer_train[
        ["Age",
         "Fare",
         "Sex",
         "Pclass",
         "IsAlone",
         "Embarked",
         "Title",
         "Age*Class"]
    ].values

    test_features = engineer_test[
        ["Age",
         "Fare",
         "Sex",
         "Pclass",
         "IsAlone",
         "Embarked",
         "Title",
         "Age*Class"]
    ].values

    target = engineer_train["Survived"].values

    forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100)
    forest.fit(train_features, target)

    predictions = forest.predict(test_features)
    passengers = engineer_test["PassengerId"].values

    data = [["PassengerId", "Survived"]]
    for index in range(0, len(passengers)):
        data.append([passengers[index], predictions[index]])

    pandas.DataFrame(data).to_csv('output/random_forest.csv', index=False, header=0)
