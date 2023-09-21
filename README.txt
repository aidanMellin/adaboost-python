Started off with the classification and writing the train program.

The train program was straighforward because you just call the requisite programs. That said with a decision that will be explained later, the hypothesisOut chosen
by whoever is running the program will be overwritten with dtClassify.py in the "training/" directory.
 
I started with decision tree because duh, you need DT to make ADA. With this, I decided to use some old hints from stack overflow and instead of writing complex JSON files
and whatnot I can simply write to a CSV, which has integration with python via pandas.

This means that I could write to binary values determined by the program with the given input and check against the test.dat files. (the bin_lang csv files in trainingData/)
I didn't end up using the other files besides train_master.dat, train.dat for training purposes, but kept the files just because they're handy.

For the dealing with DT determinations, I actually had it write to a new pythonFile for classification such that the method could be called when running the predict to 
get the data into a dictionary for pretty printing and speed. (The formatting is a bit messed up but it seems to work) -- this was fixed with a \t instead of 4 spaces

For both ada and dt math, I actually ended up using the equations listed in the wikipedia article for each subject

langFeatures.py has lists of most of the common prefixes, suffixes, and words in dutch and english, and has most of the methods for getting the data associated with inputs

Admittedly im still not too solid on adaboost, so I was mostly developing blind here. Even with what im pretty solid on submitting, the adaboost train still reports
an error rate of ~50% (using the equation), so Im not too sure what that's about, yet the predict seems to mostly function fine if not perfectly.

Depth wise, 8 iterations seemed to be where the graph starts to "level out" while retaining speed.
Most of the decision making simply comes from the common word arrays as defined in the langFeatures that then informs the decision making process based on the 
weights given to the words. In order to save time processing, the arrays were hardcoded after compiling data from txt files and lists on wikipedia 
(https://en.wikipedia.org/wiki/Most_common_words_in_English)

Example ways of running the program:

Run with DT:
python3 train.py trainingData/train.dat training/dtClassify.py dt  

Run with Ada:
python3 train.py trainingData/train.dat trainingData/adaboostclassifier.txt ada

Predict:
python3 predict.py trainingData/bin_lang_validate.csv trainingData/test2.dat  

Included in this folder is a runProgram.sh file that will execute both train/predict using dt and adaboost with the train.dat file and test on the test2.dat file
for convenience sake.

There is a verbose flag within Ada for printing between each step where the weights are calculated. 
simply adding verbose as the last argument for the train with ada will enable this (by default it is enabled in the runProgram.sh)