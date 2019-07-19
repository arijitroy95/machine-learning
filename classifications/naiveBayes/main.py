import pandas as pd
import numpy as np
import json
from collections import defaultdict


def read_csv(dataPath):
	with open(dataPath, 'r') as fp:
		dataTxt = fp.readlines()
	
	dataList = []
	for i in dataTxt:
		try:
			dataList.append(list(map(float, i.strip().split(','))))
		except ValueError:
			pass	
	return np.array(dataList)
	

def data_loading(dataPath):
	return pd.read_csv(dataPath, header=None, names=["feature1", "feature2", "outcome"])


def group_count(df):
	outputJson = defaultdict(dict)
	num_classes = df["outcome"].nunique()
	means = df.groupby('outcome').mean().values.tolist()
	variances = df.groupby('outcome').var().values.tolist()

	for i in range(num_classes):
		outputJson["class_" + str(i)]["count"] = int(df["outcome"][df["outcome"] == i].count())	
		outputJson["class_" + str(i)]["mean"] = means[i]
		outputJson["class_" + str(i)]["var"] = variances[i]
	outputJson["total"]["count"] = int(df["outcome"].count())
	with open('training.json', 'w+') as fp:
		json.dump(outputJson, fp, indent=4, sort_keys=True)
	
	

def prob_x_given_y(x, mean_xy, var_xy):
	p = (1 / (np.sqrt(2 * np.pi * var_xy))) * (np.exp((-(x - mean_xy) ** 2) / (2 * var_xy)))
	return p


def train(dataPath):
	df = data_loading(dataPath)
	group_count(df)


def execute(x):
	with open('training.json', 'r') as fp:
		dataCount = json.load(fp)
	totalCount = dataCount["total"]["count"]
	num_classes = len(dataCount.keys()) - 1
	
	class_probs = []
	for i in range(num_classes):
		prob = dataCount["class_" + str(i)]["count"] / totalCount
		class_probs.append(prob)
	
	outProb = []
	for classNum, prob in enumerate(class_probs):
		classData = dataCount["class_" + str(classNum)]
		temp = [prob_x_given_y(x[i], classData["mean"][i], classData["var"][i]) for i in range(len(x))]
		classProb = prob * np.prod(temp)
		outProb.append(classProb)
	return np.argmax(np.array([outProb]))


def accuracy(y_pred, y):
	y_new = y_pred
	y = np.ravel(y).tolist()
	TP, FN, FP, TN = 0, 0, 0, 0
	for idx, i in enumerate(y_new):
		if (i == 1) and (y[idx] == 1):
			TP += 1
		if (i == 1) and (y[idx] == 0):
			FP += 1
		if(i == 0) and (y[idx] == 1):
			FN += 1
		if(i == 0) and (y[idx] == 0):
			TN += 1
	return [[TP, FN], [FP, TN]]
		

def main():
	dataPath = "dataset.csv"
	train(dataPath)
	dataList = read_csv(dataPath)
	X = dataList[:, :-1].tolist()
	y = dataList[:, -1]
	y_pred = [execute(i) for i in X]
	print(accuracy(y_pred, y))
	print(execute([5.2, 1.2]))


if __name__ == "__main__":
	main()
