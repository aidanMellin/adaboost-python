# Library for loading a csv file and converting
# it into a dictionary
import pandas as pd
import csv
def dtPredict(csvFile):
	datas = pd.read_csv(csvFile)
	TP = 0
	TN = 0
	with open('trainingData/classifications.csv', 'w') as resultantCSV:
		writer = csv.writer(resultantCSV)
		writer.writerow(['result'])
		for dataRecord in range(datas.shape[0]):
			if datas.iloc[dataRecord]['VowelComboDutch'] <= 0:
				Lang = 0
			else:
				Lang = 1
			result = 'en' if Lang == 1 else 'nl'
			print(result)
			writer.writerow([Lang])
			if Lang == datas.iloc[dataRecord]['Lang']:
				if  Lang == 0:
					TN += 1
				else:
					TP += 1

		resultantCSV.close()
		print('TP: ' + str(TP) + ' TN: ' + str(TN))
		accuracy = (TP + TN)/datas.shape[0]
		print('Accuracy: ' + str(accuracy))

if __name__ == '__main__':
	dtPredict("trainingData/bin_lang_validate.csv")