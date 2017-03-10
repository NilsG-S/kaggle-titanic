import pandas as pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


FEATURES = [
    "Age",
    "Fare",
    "Sex",
    "Pclass",
    "IsAlone",
    "Embarked",
    "Title",
    "Age*Class"
]


def grid():
    engineer_train = pandas.read_csv(
        'engineered/engineer_train.csv', header=0
    )

    train_features = engineer_train[FEATURES].values
    target = engineer_train["Survived"].values

    grid_forest = RandomForestClassifier()

    parameters = {
        'n_estimators': [4, 6, 9],
        'max_features': ['log2', 'sqrt', 'auto'],
        'criterion': ['entropy', 'gini'],
        'max_depth': [2, 3, 5, 10],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 5, 8]
    }

    accuracy = make_scorer(accuracy_score)

    grid_search = GridSearchCV(grid_forest, parameters, scoring=accuracy)
    grid_search.fit(train_features, target)

    print(grid_search.best_params_)


def test():
    engineer_train = pandas.read_csv(
        'engineered/engineer_train.csv', header=0
    )

    train_features = engineer_train[FEATURES].values

    x_train = train_features[0:445]
    y_train = train_features[446::]

    x_target = engineer_train["Survived"].values[0:445]
    y_comp = engineer_train["Survived"].values[446::]

    test_forest = RandomForestClassifier(n_estimators=100)
    test_forest.fit(x_train, x_target)
    test_pred = test_forest.predict(y_train)

    correct = 0
    for i in range(0, y_comp.size):
        if y_comp[i] == test_pred[i]:
            correct += 1

    print(correct / y_comp.size)


def learn():
    engineer_train = pandas.read_csv(
        'engineered/engineer_train.csv', header=0
    )
    engineer_test = pandas.read_csv(
        'engineered/engineer_test.csv', header=0
    )

    train_features = engineer_train[FEATURES].values

    test_features = engineer_test[FEATURES].values

    target = engineer_train["Survived"].values

    forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100)
    forest.fit(train_features, target)

    predictions = forest.predict(test_features)
    passengers = engineer_test["PassengerId"].values

    data = [["PassengerId", "Survived"]]
    for index in range(0, len(passengers)):
        data.append([passengers[index], predictions[index]])

    pandas.DataFrame(data).to_csv('output/random_forest.csv', index=False, header=0)
