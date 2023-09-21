
# --- imports -----
import math
import pandas as pd
import numpy as np

# Methods from language features
from training.langFeatures import *

survBinary = None
target = None
outFile = None
DEPTH = 1
WEIGHT = None


class LeafNode:
    """
    Holds a dict of class (eg. "vowelDutch") -> # of times
    it appears in the rows from the training data that reach this LeafNode.
    """

    def __init__(self, record_data, attribute, target_name, is_left, conditionalStructure):
        self.record_data = record_data
        self.attribute = attribute
        self.target_name = target_name
        self.is_left = is_left
        self.conditionalStructure = conditionalStructure
        self.predictions = {1: len(partition(record_data, "Lang")[0]), 0: len(
            partition(record_data, "Lang")[1])}


class DecisionNode:
    """
    A Decision node asks a question. It holds a ref to the
    Q, and to the two children nodes
    """

    def __init__(self, attribute, tBranch, fBranch, depth):
        self.attribute = attribute
        self.tBranch = tBranch
        self.fBranch = fBranch
        self.depth = depth


def entropy(counter1, counter2):
    """
    Calculate the entropy of the two counts

    Param:
        counter1 - # of counts in set 1
        counter2 - # of counts in set 2
    """

    vEntropy = 0.0

    if counter1 != 0 or counter2 != 0:
        # The total number of values
        total = counter1 + counter2

        # Log(0) returns an error, so errors are handled

        if counter1 == 0 and counter2 == 0:  # If both counts == 0, then the result is whack
            vEntropy = 1.0
        # If one of the counts = 0, consider the count[0] = 0
        elif counter1 == 0:
            vEntropy = -(0 + (counter2 / total) *
                         math.log2((counter2 / total)))
        # If one of the counts = 1, consider the count[1] = 1
        elif counter2 == 0:
            vEntropy = -((counter1 / total) *
                         math.log2((counter1 / total)) + 0)
        else:  # None of the counts are 0
            vEntropy = -((counter1 / total) * math.log2((counter1 / total)) +
                         (counter2 / total) * math.log2((counter2 / total)))

    # Using the vEntropy above, if it is a 0, it represents as a -0.0 (fixed this with conditional return)
    if vEntropy == -0.0:
        return 0.0
    else:
        return vEntropy


def writeHead():
    """
    Open up a file pointer to a file named classifications
    """
    # Write a line at the end of the file

    # Write a program that writes the program
    outFile.write("# Library for loading a csv file and converting\n"
                  "# it into a dictionary\n"
                  "import pandas as pd\n"
                  "import csv\n"
                  "def dtPredict(csvFile):\n"
                  "\tdatas = pd.read_csv(csvFile)\n"
                  "\tTP = 0\n"
                  "\tTN = 0\n"
                  "\twith open('trainingData/classifications.csv', 'w') as resultantCSV:\n"
                  "\t\twriter = csv.writer(resultantCSV)\n"
                  "\t\twriter.writerow(['result'])\n"
                  "\t\tfor dataRecord in range(datas.shape[0]):\n")


def writeClassify():
    """
    This will create the main function in the classifier
    It creates the classifier program
    It writes the result to the CSV
    It calculates the prediction accuracy on Validation data
    """

    # Build tree
    decTree = buildTree(survBinary, target, 0, False, "")

    # print the resulting tree to the classifier program
    printTree(decTree, outFile, "\t\t\t")

    # Converting binary numbers into nl and en values
    convert = "\t\t\tresult = 'en' if " + target + " == 1 else 'nl'\n"

    # Write the converter to file
    outFile.write(convert)

    # Print prediction output
    # If Classifier thinks it is English, print 1 else print 0
    outFile.write("\t\t\tprint(result)\n")

    # Write the result to csv file
    outFile.write("\t\t\twriter.writerow([" + target + "])\n")

    # Get statistics value for classifier
    # Calculate the TP and TN
    outFile.write(f"\t\t\tif {target} == datas.iloc[dataRecord]['{target}']:\n"
                  f"\t\t\t\tif  {target} == 0:\n"
                  "\t\t\t\t\tTN += 1\n"
                  "\t\t\t\telse:\n"
                  "\t\t\t\t\tTP += 1\n\n"
                  "\t\tresultantCSV.close()\n"
                  "\t\tprint('TP: ' + str(TP) + ' TN: ' + str(TN))\n"
                  "\t\taccuracy = (TP + TN)/datas.shape[0]\n"
                  "\t\tprint('Accuracy: ' + str(accuracy))\n")

    return decTree


def writeTail(csvFile):
    outFile.write("\nif __name__ == '__main__':\n\tdtPredict(\"" + csvFile + "\")")\


def partition(recData, attribute):
    """
    Partition the dataset

    For each row in the dataset, check if it matches the attribute. If so, add it to the 'true rows', otherwise, add it to 'false rows'
    """

    # if the record for that attribute is 1, it is true else false
    tRows = recData[recData[attribute] == 1]
    fRows = recData[recData[attribute] == 0]

    return tRows, fRows


def findSplit(recData, target):
    """
    Find the best Q by iterating over every attrib and calculating min info gain
    """
    attribs = recData.keys()
    # Keep track of the best attrib that produces the best entropy
    bestAttrib = attribs[0]
    nAttribs = recData.shape[1]
    minInfoGain = float("inf")  # Storing the min entropy
    bestSplit = None
    conditionalStructure = None

    for col in range(nAttribs):  # For each attrib
        if attribs[col] == target:
            continue

        # Get the name of attrib
        attrib = attribs[col]

        attrib_true = recData[attrib] == 1  # attrib == True
        target_true = recData[target] == 1  # target variable == True
        attrib_false = recData[attrib] == 0  # attrib == False
        target_false = recData[target] == 0  # target variable == False

        # attrib == True, Target == True
        truePosRows = recData[attrib_true & target_true]
        # attrib == False, Target == False
        trueNegRows = recData[attrib_false & target_false]
        # attrib == True, Target == False
        falsePosRows = recData[attrib_true & target_false]
        # attrib == False, Target == True
        falseNegRows = recData[attrib_false & target_true]

        falseNegCount = falseNegRows.shape[0]  # False negative count
        falsePosCount = falsePosRows.shape[0]  # False positive count
        truePosCount = truePosRows.shape[0]  # True positive count
        trueNegCount = trueNegRows.shape[0]  # True negative count

        # Get the best between false and true postive/negative
        true_max = max(truePosCount, falsePosCount)
        false_max = max(falseNegCount, trueNegCount)

        if true_max >= false_max:
            if truePosCount >= falsePosCount:
                conditionalStructure = "true_true"
            else:
                conditionalStructure = "true_false"
        else:
            if falseNegCount >= trueNegCount:  # attrib == False, target == True
                conditionalStructure = "false_true"
            else:  # attrib == False, target == False
                conditionalStructure = "false_false"

        if WEIGHT is not None:  # weight is specified

            falseNegCount = 0
            falsePosCount = 0
            truePosCount = 0
            trueNegCount = 0

            # Original calc = Calculate W*P(x|tp)*log2(P(x|tp))
            # Traverse the record and find/calc weighted entropy
            for index in range(recData.shape[0]):

                # True positive
                if recData.iloc[index][attrib] == 1 and recData.iloc[index][target] == 1:
                    truePosCount += WEIGHT[index]

                # True negative
                elif recData.iloc[index][attrib] == 0 and recData.iloc[index][target] == 0:
                    trueNegCount += WEIGHT[index]

                # False Positive
                elif recData.iloc[index][attrib] == 1 and recData.iloc[index][target] == 0:
                    falsePosCount += WEIGHT[index]

                # False Negative
                elif recData.iloc[index][attrib] == 0 and recData.iloc[index][target] == 1:
                    falseNegCount += WEIGHT[index]

                true_max = max(truePosCount, falsePosCount)
                false_max = max(falseNegCount, trueNegCount)

                if true_max >= false_max:
                    if truePosCount >= falsePosCount:  # if attrib == True, target == True
                        conditionalStructure = "true_true"
                    else:  # attrib == True, target == False
                        conditionalStructure = "true_false"
                else:
                    if falseNegCount >= trueNegCount:  # if attrib == False, target == True
                        conditionalStructure = "false_true"
                    else:  # attrib == False, target == False
                        conditionalStructure = "false_false"

        # If current entropy is smaller than min, update the entropy, and note the best split point
        false_entropy = entropy(falseNegRows.shape[0], trueNegRows.shape[0])
        true_entropy = entropy(truePosRows.shape[0], falsePosRows.shape[0])

        total = falseNegCount + falsePosCount + truePosCount + trueNegCount

        if total == 0:
            continue

        # Mix entropy from left/right node
        newMinInfo = ((falseNegCount + trueNegCount) / total) * false_entropy + \
            ((falsePosCount + truePosCount) / total) * true_entropy

        if newMinInfo < minInfoGain:  # If current entropy is smaller than min, update the entropy, and best attrib also note the best split point
            minInfoGain, bestAttrib = newMinInfo, attrib
            bestSplit = (partition(recData, attrib))

    return minInfoGain, bestAttrib, bestSplit, conditionalStructure


def buildTree(recData, target_name, depth, is_left, conditionalStructure_p):
    """
    Recursive tree build
    """

    min_info_gain, attribute, best_split, conditionalStructure = findSplit(
        recData, target_name)

    if depth == DEPTH or best_split is None:
        if conditionalStructure_p == "":
            return LeafNode(recData, attribute, target_name, is_left, conditionalStructure)
        else:
            return LeafNode(recData, attribute, target_name, is_left, conditionalStructure_p)

    best_right = best_split[0]  # Best True branch at this instance
    best_left = best_split[1]  # Best False Branch at this instance

    tBranch = buildTree(best_right, target_name, depth +
                        1, False, conditionalStructure)
    fBranch = buildTree(best_left, target_name, depth +
                        1, True, conditionalStructure)
    return DecisionNode(attribute, tBranch, fBranch, depth)


def dtPredict(node, datas, single=False):
    """
    Main function to predict if
    it is eng or dutch given var
    """
    predictions = []

    if single:
        return dtPredictHelp(node, datas)
    else:
        for dataRecord in range(datas.shape[0]):
            predictions.append(dtPredictHelp(node, datas.iloc[dataRecord]))

        return np.array(predictions)


def dtPredictHelp(node, dataRecord):
    """
    Main function for predicting
    """
    if isinstance(node, LeafNode):
        if node.is_left:
            if node.conditionalStructure == "true_true" or node.conditionalStructure == "false_false":
                return 0
            elif node.conditionalStructure == "true_false" or node.conditionalStructure == "false_true":
                return 1
        else:
            if node.conditionalStructure == "true_true" or node.conditionalStructure == "false_false":
                return 1
            elif node.conditionalStructure == "true_false" or node.conditionalStructure == "false_true":
                return 0
        return

    if dataRecord[node.attribute] <= 0:
        return dtPredictHelp(node.fBranch, dataRecord)
    else:
        return dtPredictHelp(node.tBranch, dataRecord)


def dtFit(examples="trainingData/train.dat", hypothesisOut="training/dtClassify.py", depth=2, weight=None):

    # Convert example file into a csv file
    writeToCSV("trainingData/bin_lang.csv", examples)

    # Load Data
    global survBinary
    survBinary = pd.read_csv('trainingData/bin_lang.csv')

    global target
    target = "Lang"

    global DEPTH
    DEPTH = depth

    global WEIGHT
    WEIGHT = weight

    global outFile
    outFile = open(hypothesisOut, "w")

    # write header = Include import functions
    writeHead()

    # Create the classifier
    decTree = writeClassify()

    # Call the classifier function and get validation accuracy
    writeTail("trainingData/bin_lang_validate.csv")

    # Close Classifier file
    outFile.close()

    return decTree


def printTree(node, outFile, spacing=""):

    # Base case: we've reached a LeafNode
    if isinstance(node, LeafNode):
        # If the number of english is greater than the number of dutch, it is eng
        if node.predictions[0] >= node.predictions[1]:
            outFile.write(f"{spacing}{target} = 0\n")
        else:
            outFile.write(f"{spacing}{target} = 1\n")
        return

    # Print Question
    outFile.write(
        f"{spacing}if datas.iloc[dataRecord]['{str(node.attribute)}'] <= 0:\n")

    # false brance recursion
    printTree(node.fBranch, outFile, spacing + "\t")

    outFile.write(spacing + "else:\n")
    # true branch recursion
    printTree(node.tBranch, outFile, spacing + "\t")


def printTreeOut(node, spacing=""):
    if isinstance(node, LeafNode):
        return node.predictions

    # Print Q
    print(f"{spacing}if datas.iloc[dataRecord]['{str(node.attribute)}'] <= 0:")

    # Call recursively on false branch
    printTreeOut(node.fBranch, spacing + "\t")
    print(f"{spacing}else:")
    # Call recursively on true branch
    printTreeOut(node.tBranch, spacing + "\t")


def cleanFile(recData, target):
    """
    remove columns that aren't in binary from the CSV
    """
    modRecords = recData

    for attribute in recData.keys():
        check = recData[attribute].iloc[0]
        if check != 0 and check != 1:
            modRecords = modRecords.drop(attribute, axis=1)

    return modRecords


dtFit(depth=2)
