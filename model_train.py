from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import tensorflow as tf
import pickle
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

def svmModel(X, Y):
    print("SVM Model Start")
    SVM = svm.SVC()
    SVM.fit(X, Y)
    f = open("models/svm.pkl", "wb")
    pickle.dump(SVM, f)


def randomForestModel(X, Y):
    print("Random Forest Model Start")
    rf = RandomForestClassifier(max_depth=50)
    rf.fit(X, Y)
    f = open("models/randomforest.pkl", "wb")
    pickle.dump(rf, f)

def mlpModel(X, Y):
    print("MLP Model Start")
    classifier = MLPClassifier(hidden_layer_sizes=(
        150, 100, 50), max_iter=300, activation='relu', solver='adam', random_state=1)
    classifier.fit(X, Y)

    f = open("models/mlp.pkl", "wb")
    pickle.dump(classifier, f)

def adaboost(X,Y):
    print("Ada Boost Model Start")
    clf = AdaBoostClassifier(n_estimators=100, algorithm='SAMME',learning_rate=0.001, random_state = 1)
    clf.fit(X,Y)
    f = open("models/adaboost.pkl", "wb")
    pickle.dump(clf, f)

def gboost(X,Y):
    print("Grid Boost Model Start")
    clf = GradientBoostingClassifier(n_estimators=300, criterion='mse',max_features='auto',learning_rate=0.001, loss = 'deviance', random_state = 1)
    clf.fit(X,Y)
    f = open("models/gboost.pkl", "wb")
    pickle.dump(clf, f)

def knearestNeighboursModel(X, Y):
    print("KNN Model Start")
    knr = KNeighborsClassifier()
    knr.fit(X, Y)
    f = open("models/knn.pkl", "wb")
    pickle.dump(knr, f)

