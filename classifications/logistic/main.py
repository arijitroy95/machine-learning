import numpy as np
import json
import os
from matplotlib import pyplot as plt


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


def normalize(X):
	if os.path.exists('config.json'):
		with open('config.json', 'r') as fp:
			val = json.load(fp)
		MAX = np.array(val['max'])
		rng = np.array(val['range'])
	else:
		MIN = np.min(X, axis=0)
		MAX = np.max(X, axis=0)
		
		rng = MAX - MIN
		val = {'max': list(MAX), 'range': list(rng)}
		with open('config.json', 'w+') as fp:
			json.dump(val, fp, indent=4, sort_keys=True)
	normX = 1 - ((MAX - X) / rng)
	return normX


def sigmoid(X, beta):
	return 1 / (1 + np.exp(-np.dot(X, beta.T)))


def gradiant(X, y, beta):
	temp = sigmoid(X, beta) - y.reshape(X.shape[0], -1)
	return np.dot(temp.T, X)


def cost_func(beta, X, y):
	log_func = sigmoid(X, beta)
	y = np.squeeze(y)
	step1 = y * np.log(log_func)
	step2 = (1 - y) * np.log(1 - log_func)
	final = -step1 - step2
	return np.mean(final)


def gradiant_desc(X, y, beta, lr=0.01, epsilon=1e-3):
	cost = cost_func(beta, X, y)
	change_cost = 1
	num_iter = 1
	all_cost = []
	while change_cost > epsilon:
		old_cost = cost
		beta -= (lr * gradiant(X, y, beta))
		cost = cost_func(beta, X, y)
		change_cost = abs(cost - old_cost)
		num_iter += 1
		if num_iter % 100 == 0:
			print("Itteration: ", num_iter - 1, "\tCost: ", cost)
		all_cost.append(cost)
	plt.plot(all_cost, c='b')
	plt.show()
	return beta, num_iter


def pred_values(X, beta, th=0.5):
	pred_prob = sigmoid(X, beta)
	pred_value = np.where(pred_prob >= th, 1, 0)
	
	pred_prob = np.ravel(pred_prob)
	pred_value = np.ravel(pred_value)
	finalAns = []

	for p, v in zip(pred_prob, pred_value):
		if int(v) == 0:
			finalAns.append({"value": int(v), "prob": 1 - float(p)})
		else:
			finalAns.append({"value": int(v), "prob": float(p)})
	return finalAns


def plot_reg(X, y, beta):
	x_0 = X[np.where(y == 0.0)]
	x_1 = X[np.where(y == 1.0)]
	
	plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='r', label='y = 0')
	plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='g', label='y = 1')
	
	x1 = np.arange(0, 1, 0.1)
	x2 = - (beta[0, 0] + beta[0, 1] * x1) / beta[0, 2]
	
	plt.plot(x1, x2, c='k', label='dicission boundry')
	
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.legend()
	plt.show()


def accuracy(y_pred, y):
	y_new = [i["value"] for i in y_pred]
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


def dataLoading(dataPath):
	dataset = read_csv(dataPath)
	X = normalize(dataset[:, :-1])
	X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))
	y = dataset[:, -1]
	return X, y


def train(dataPath):
	X, y = dataLoading(dataPath)
	beta = np.matrix(np.zeros(X.shape[1]))
	
	beta, num_iter = gradiant_desc(X, y, beta)
	plot_reg(X, y, beta)
	np.save('beta_values.npy', beta)


def execute(x, norm=False):
	if not norm:
		x = np.matrix(x)
		x = normalize(x)
		x = np.hstack((np.matrix(np.ones(x.shape[0])).T, x))
	
	beta = np.load('beta_values.npy')
	y_pred = pred_values(x, beta)
	return y_pred

def main():
	dataPath = "dataset.csv"
	train(dataPath)
	X, y = dataLoading(dataPath)
	y_pred = execute(X, norm=True)
	print(accuracy(y_pred, y))
	y_test = execute([7, 5])
	print(json.dumps(y_test, indent=4))
	


if __name__ == "__main__":
	main()
