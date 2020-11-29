from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import pickle

def svmModel(X, Y):
	n = len(X)
	p = int(0.7 * n)
	print("SVM Model Start")
	SVM = svm.SVC()
	SVM.fit(X[:p], Y[:p])
	prediction = SVM.predict(X[p:])
	accuracy = accuracy_score(prediction, Y[p:])
	print("SVM Accuracy: ", accuracy)

	f = open("models/svm.pkl", "wb")
	pickle.dump(SVM, f)

def randomForestModel(X, Y):
	n = len(X)
	p = int(0.7 * n)
	rf = RandomForestClassifier(max_depth=50)
	rf.fit(X[:p], Y[:p])
	prediction = rf.predict(X[p:])
	accuracy = accuracy_score(prediction, Y[p:])
	print("Random Forest Regression Accuracy: ", accuracy)

	f = open("models/randomforest.pkl", "wb")
	pickle.dump(rf, f)

def mlpModel(X, Y):
	n = len(X)
	p = int(0.7 * n)
	classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300, activation = 'relu', solver='adam', random_state=1)
	classifier.fit(X[:p], Y[:p])
	prediction = classifier.predict(X[p:])
	accuracy = accuracy_score(prediction, Y[p:])
	print("MLP Classifier Accuracy: ", accuracy)

	f = open("models/mlp.pkl", "wb")
	pickle.dump(classifier, f)

