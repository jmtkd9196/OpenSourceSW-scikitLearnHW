#PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/jmtkd9196/OpenSourceSW-scikitLearnHW/blob/main/OpenSourceSW-scikitLearnHW/template.py

import sys
import numpy as np										## numpy
import pandas as pd										## pandas
from sklearn.model_selection import train_test_split	## 데이터 분할
from sklearn.tree import DecisionTreeClassifier			## 의사결정나무
from sklearn.metrics import accuracy_score, precision_score, recall_score            ## 정확도, 정밀도, 재현율
from sklearn.ensemble import RandomForestClassifier		## 랜덤포레스트
from sklearn.svm import SVC								## 서포트 벡터 머신
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_dataset(dataset_path):
	#To-Do: Implement this function
	dataset_df = pd.read_csv(dataset_path)
	return dataset_df

def dataset_stat(dataset_df):
	#To-Do: Implement this function
	return dataset_df.shape[1] - 1, dataset_df.groupby("target").size()[0], dataset_df.groupby("target").size()[1]

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	x_train, x_test, y_train, y_test = train_test_split(dataset_df.drop('target', axis=1), dataset_df.target, test_size=testset_size)
	return x_train, x_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)
	return accuracy_score(y_test, dt_cls.predict(x_test)), precision_score(y_test, dt_cls.predict(x_test)), recall_score(y_test, dt_cls.predict(x_test))

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)
	return accuracy_score(y_test, rf_cls.predict(x_test)), precision_score(y_test, rf_cls.predict(x_test)), recall_score(y_test, rf_cls.predict(x_test))

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	svm_pipe = make_pipeline(StandardScaler(), SVC())
	svm_pipe.fit(x_train, y_train)
	return accuracy_score(y_test, svm_pipe.predict(x_test)), precision_score(y_test, svm_pipe.predict(x_test)), recall_score(y_test, svm_pipe.predict(x_test))

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)