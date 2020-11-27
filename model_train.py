from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def svmModel(X, Y):
	n = len(X)
	p = int(0.7 * n)
	print("SVM Model Start")
	SVM = svm.SVC()
	SVM.fit(X[:p], Y[:p])
	prediction = SVM.predict(X[p:])
	accuracy = accuracy_score(prediction, Y[p:])
	print("SVM Accuracy: ", accuracy)

def randomForestModel(X, Y):
	n = len(X)
	p = int(0.7 * n)
	rf = RandomForestClassifier()
	rf.fit(X[:p], Y[:p])
	prediction = rf.predict(X[p:])
	accuracy = accuracy_score(prediction, Y[p:])
	print("Random Forest Regression Accuracy: ", accuracy)