import numpy as np
import json
import random
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
	return dataList


def train(dataPath):
	data = read_csv(dataPath)

	with open("trainingData.json", "w+") as fp:
		json.dump(data, fp)


def find_distance(x1, x2):
	return np.linalg.norm(np.array(x1) - np.array(x2))


def accuracy(y_new, y):
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


def execute(x_new, K=3): 
	with open("trainingData.json", "r") as fp:
		dataPts = json.load(fp)
	
	dist_and_class = list()
	for data in dataPts:
		x = data[:-1]
		y = int(data[-1])
		dist = find_distance(x, x_new)
		dist_and_class.append((dist, y))
	
	sorted_dist = sorted(dist_and_class, key=lambda x: x[0])
	vote_of_k = sorted_dist[: K]
	tempAns = defaultdict(int)
	for i in vote_of_k:
		tempAns[i[1]] += 1
	finalAns = sorted(tempAns.items(), key=lambda x: x[1])
	return finalAns[0][0]


def main():
	dataPath = "dataset.csv"
	train(dataPath)
	dataList = read_csv(dataPath)
	y_pred = [execute(i[:-1]) for i in dataList]
	y_acc = [int(i[-1]) for i in dataList]
	print(accuracy(y_pred, y_acc))
	print(execute([0, 0]))
	

if __name__ == "__main__":
	main()
