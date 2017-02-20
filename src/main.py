import cleaning
import modeling
import testing
import engineering
import learning


# Old

def call_gen_model():
    modeling.gen_model()


def call_test():
    testing.test()


# New

def call_clean():
    cleaning.clean('../data/train.csv', '../cleaned/clean_train.csv')
    cleaning.clean('../data/test.csv', '../cleaned/clean_test.csv')

    return True


def call_gen_features():
    engineering.gen_features(
        '../cleaned/clean_train.csv',
        '../engineered/engineer_train.csv'
    )
    engineering.gen_features(
        '../cleaned/clean_test.csv',
        '../engineered/engineer_test.csv'
    )

    return True


def call_learn():
    learning.learn()

    return True


def call_exit():
    return False


def main():
    control = True
    switch = {
        "clean": call_clean,
        "gen_features": call_gen_features,
        "learn": call_learn,
        "exit": call_exit
    }

    while control:
        command = input("Enter a command: ")

        control = switch[command]()

main()
