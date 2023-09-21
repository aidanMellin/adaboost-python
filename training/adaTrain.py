from training.dtClassify import *
from training.dtTrain import *
from training.langFeatures import *
import numpy as np


def adaTrainMain(csvFile="trainingData/bin_lang.csv", countIteration=3, verbose=False):
    """
    csvFile - the file with the training data
    """
    # Parse the csv file as data
    datas = pd.read_csv(csvFile)

    y = datas["Lang"]

    datas = datas.drop(["Lang"], axis=1)

    N, _ = datas.shape

    if verbose:
        print(f"Num Samples:\t{N}")

    weights = np.ones(N) / N  # Initialise the weights as 1/N
    stumpValue = []
    hypWeights = [1] * countIteration

    for t in range(countIteration):
        print()
        # Generate a decision stump given then the weight
        h = dtFit(depth=1, weight=weights)

        # Predict using the decision stump generated
        pred = dtPredict(h, datas)

        error = 0

        for i in range(len(pred)):
            if pred[i] != y.iloc[i]:
                error = error + weights[i]
            else:
                weights[i] = weights[i] * error / (1 - error)

        total = 0
        # Normalize weights
        for weight in weights:
            total += weight

        for i in range(N):
            weights[i] = weights[i] / total

        # Update the hypothesis weight
        hypWeights[t] = math.log(((1 - error) / (error)), 2)

        # Append the stump
        stumpValue.append(h)

        if verbose:
            print(
                f"Iteration: {t}\nBest Attribute:\t{h.attribute}\nMod Weights:\n{weights}\nError Rate:\t{error}")

    return hypWeights, stumpValue


def adaPredict(test_csv, hypWeights, stumpValue):
    '''
    Make a prediction and calculate the
    accuracy
    '''
    # Parse csv file
    tests = pd.read_csv(test_csv)

    # Save target variable
    y = tests["Lang"]

    tests = tests.drop(["Lang"], axis=1)

    summed = 0
    correct = 0
    incorrect = 0

    '''
    Traverse through every test. If the prediction
    was 0, negate the summed. If the prediction
    was 1, increase the weight 
    '''
    for testIdx in range(tests.shape[0]):
        for stumpIdx in range(len(stumpValue)):
            # Make the prediction using the saved decision stump
            pred = dtPredict(stumpValue[stumpIdx],
                             tests.iloc[testIdx], single=True)
            if pred == 0:
                summed += -1 * hypWeights[stumpIdx]
            elif pred == 1:
                summed += 1 * hypWeights[stumpIdx]

        if summed > 0:
            if y.iloc[testIdx] == 1:
                correct += 1
            else:
                incorrect += 1
        else:
            if y.iloc[testIdx] == 0:
                correct += 1
            else:
                incorrect += 1

    return correct/y.shape[0]


def adaComplete(examples="training/train.txt", hypothesisOut="adaBoostClassifier.py", verbose=False):
    '''
    The main function that
    does all of the ada-boosting
    training
    '''
    hypWeights, stumpValue = adaTrainMain(countIteration=8, verbose=verbose)

    with open(hypothesisOut, 'w') as resultantCSV:
        writer = csv.writer(resultantCSV)
        writer.writerow(['hypWeights', 'stumpValue'])
        writer.writerow([hypWeights, stumpValue])
        resultantCSV.close()

    # Convert example file into a csv file
    writeToCSV("trainingData/bin_lang.csv", examples)
    print(adaPredict("trainingData/bin_lang_validate.csv", hypWeights, stumpValue))
