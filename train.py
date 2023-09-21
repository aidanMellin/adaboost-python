import sys

# Import methods from training library
from training.dtTrain import *
from training.adaTrain import *


def main():
    if len(sys.argv) <= 3:
        print('Usage: {} <examples> <hypothesisOut> <learning-type>'.format(
            sys.argv[0]))
        exit(1)

    examples = sys.argv[1]
    hypothesisOut = sys.argv[2]
    learningType = sys.argv[3]
    verbose = False
    if (len(sys.argv) == 5):  # Verbose printing on
        if (sys.argv[4] == "verbose"):
            verbose = True

    if learningType == "dt":  # Train using DT
        dtFit(examples=examples, hypothesisOut="training/dtClassify.py")
    elif learningType == "ada":  # Train with ada
        adaComplete(examples=examples,
                    hypothesisOut=hypothesisOut, verbose=verbose)


if __name__ == '__main__':
    main()
