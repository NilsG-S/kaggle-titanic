import cleaning
import modeling
import testing
import engineering


def main():
    cleaning.clean('../data/train.csv', '../cleaned/clean_train.csv')
    cleaning.clean('../data/test.csv', '../cleaned/clean_test.csv')

    modeling.gen_model()

    engineering.gen_features(
        '../cleaned/clean_train.csv',
        '../engineered/engineer_train.csv'
    )
    engineering.gen_features(
        '../cleaned/clean_test.csv',
        '../engineered/engineer_test.csv'
    )

    testing.test()

main()
