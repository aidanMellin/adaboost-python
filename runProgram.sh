python3 -m pip install -r requirements.txt

echo "\n----Train and Predict with DT----\n"
python3 train.py trainingData/train.dat training/dtClassify.py dt
echo "\npredict\n"
python3 predict.py trainingData/bin_lang_validate.txt trainingData/test2.dat  

echo "\n----Train and Predict with ADA----\n"
python3 train.py trainingData/train.dat trainingData/adaboostclassifier.txt ada verbose
echo "\npredict\n"
python3 predict.py trainingData/bin_lang_validate.txt trainingData/test2.dat  