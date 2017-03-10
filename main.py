import src.cleaning as cleaning
import src.engineering as engineering
import src.learning as learning


# New

def call_clean():
    cleaning.clean('data/train.csv', 'cleaned/clean_train.csv')
    cleaning.clean('data/test.csv', 'cleaned/clean_test.csv')

    return True


def call_gen_features():
    engineering.gen_features(
        'cleaned/clean_train.csv',
        'engineered/engineer_train.csv'
    )
    engineering.gen_features(
        'cleaned/clean_test.csv',
        'engineered/engineer_test.csv'
    )

    return True


def call_learn():
    learning.learn()

    return True

def call_test():
    learning.test()

    return True


def call_exit():
    return False


def main():
    control = True
    switch = {
        "clean": call_clean,
        "gen_features": call_gen_features,
        "learn": call_learn,
        "test": call_test,
        "exit": call_exit
    }

    while control:
        command = input("Enter a command: ")

        control = switch[command]()

main()
