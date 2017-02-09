import cleaning
import modeling
import testing


def main():
    cleaning.clean('../data/train.csv', '../cleaned/clean_train.csv')
    cleaning.clean('../data/test.csv', '../cleaned/clean_test.csv')

    modeling.gen_model()

    testing.test()

main()
