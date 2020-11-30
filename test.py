import text_feature_extraction 
import audio_feature_extraction
from lexiconFeatureVector import lexicons

import pickle
import pandas as pd
import numpy as np
from nltk.util import ngrams
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

def svmPredict(X, Y):
	f = open("models/svm.pkl", "rb")
	SVM = pickle.load(f)
	prediction = SVM.predict(X)
	accuracy = accuracy_score(prediction, Y)
	a = balanced_accuracy_score(prediction, Y)
	print("SVM Accuracy: ", accuracy)
	print("weighted: ", a)
	print(classification_report(Y, prediction))

def randomForestPredict(X, Y):
	f = open("models/randomforest.pkl", "rb")
	rf = pickle.load(f)
	prediction = rf.predict(X)
	accuracy = accuracy_score(prediction, Y)
	a = balanced_accuracy_score(prediction, Y)

	print("Random Forest Regression Accuracy: ", accuracy)
	print("weighted: ", a)
	print("RF F1 Score:", f1)
	print(classification_report(Y, prediction))

def mlpPredict(X, Y):
	f = open("models/mlp.pkl", "rb")
	classifier = pickle.load(f)
	prediction = classifier.predict(X)
	accuracy = accuracy_score(prediction, Y)
	a = balanced_accuracy_score(prediction, Y)
	f1 = f1_score(prediction, Y, average='weighted')

	print("MLP Classifier Accuracy: ", accuracy)
	print("weighted: ", a)
	print("MLP F1 Score:", f1)
	print(classification_report(Y, prediction))

def gboostPredict(X, Y):
	f = open("models/gboost.pkl", "rb")
	classifier = pickle.load(f)
	prediction = classifier.predict(X)
	accuracy = accuracy_score(prediction, Y)
	a = balanced_accuracy_score(prediction, Y)
	f1 = f1_score(prediction, Y, average='weighted')

	print("Gboost Classifier Accuracy: ", accuracy)
	print("weighted: ", a)
	print("Gboost F1 Score:", f1)
	print(classification_report(Y, prediction))

def adaboostPredict(X, Y):
	f = open("models/adaboost.pkl", "rb")
	classifier = pickle.load(f)
	prediction = classifier.predict(X)
	accuracy = accuracy_score(prediction, Y)
	a = balanced_accuracy_score(prediction, Y)
	f1 = f1_score(prediction, Y, average='weighted')

	print("Ada Boost Classifier Accuracy: ", accuracy)
	print("weighted: ", a)
	print("Ada Boost F1 Score:", f1)
	print(classification_report(Y, prediction))

def naiveBayesPredict(X, Y):
	f = open("models/naiveBayes.pkl", "rb")
	nb = pickle.load(f)
	prediction = nb.predict(X)
	accuracy = accuracy_score(prediction, Y)
	f1 = f1_score(prediction, Y, average='weighted')

	print('Naive Bayes Accuracy: ', accuracy)
	print('Naive Bayes F1 Score: ', f1)
	print(classification_report(Y, prediction))


def knearestNeighboursPredict(X, Y):
	f = open("models/knn.pkl", "rb")
	knr = pickle.load(f)
	prediction = knr.predict(X)
	accuracy = accuracy_score(prediction, Y)
	f1 = f1_score(prediction, Y, average='weighted')

	print("K nearest neighbours Accuracy: ", accuracy)
	print("K nearest neighbours f1: ", f1)
	print(classification_report(Y, prediction))
	


def ngram(utterances, utterances_tokenized):
	unigramdict = pickle.load(open("dict/unigram_dict.pkl", "rb"))
	bigramdict = pickle.load(open("dict/bigram_dict.pkl", "rb"))
	lenuni = len(unigramdict)
	lenbi = len(bigramdict)
	res1 = np.zeros([len(utterances), lenuni])
	res2 = np.zeros([len(utterances), lenbi])
	for i, tweet in enumerate(utterances):
		arr1 = np.zeros([lenuni])
		arr2 = np.zeros([lenbi])
		uni = ngrams(tweet.split(),1)
		bi = ngrams(tweet.split(),2)
		for u in uni:
			if u in unigramdict:
				arr1[unigramdict[u]] = 1
		for b in bi:
			if b in bigramdict:
				arr2[bigramdict[b]] = 1
		res1[i] = arr1
		res2[i] = arr2

	return res1, res2

if __name__ == '__main__':
	test_utterance_tokenized = []
	test_df = pd.read_csv("text_data/test_sent_emo.csv")
	# test_utterance = test_df["Utterance"].apply(preprocess).values.tolist()
	test_utterance = test_df["Utterance"].values.tolist()
	test_emo = test_df["Emotion"].values.tolist()
	test_sentiment = test_df["Sentiment"].values.tolist()
	text_feature_extraction.tokenized(test_utterance, test_utterance_tokenized)

	f = open("models/onehot.pkl", "rb")
	enc = pickle.load(f)
	test_sentiment = enc.transform(test_sentiment)

	unigramVector_test, bigramVector_test = ngram(test_utterance, test_utterance_tokenized)
	lexicon_test = lexicons(test_utterance_tokenized)
	audio_test = audio_feature_extraction.dictToarr("test")

	vector = np.zeros([len(test_utterance), len(unigramVector_test[0]) + len(bigramVector_test[0]) + 1 + len(lexicon_test[0]) + len(audio_test[0])])
	for i in range(len(test_utterance)):
		vector[i] = np.concatenate((unigramVector_test[i], bigramVector_test[i], np.array([test_sentiment[i]]) ,lexicon_test[i], audio_test[i]))

	f1 = open("models/kbest.pkl", "rb")
	f2 = open("models/scaler.pkl", "rb")
	f3 = open("models/pca.pkl", "rb")
	f4 = open("models/labelencoder.pkl", "rb")
	# f = open("models/", "rb")

	sel = pickle.load(f1)
	sc = pickle.load(f2)
	pca = pickle.load(f3)
	label_encoder = pickle.load(f4)

	vector = sel.transform(vector)
	vector = sc.transform(vector)
	vector = pca.transform(vector)

	test_emo = label_encoder.transform(test_emo)

	print(vector.shape)
	print(test_emo.shape)

	svmPredict(vector, test_emo)
	randomForestPredict(vector, test_emo)
	mlpPredict(vector, test_emo)
	# gboostPredict(vector, test_emo)
	adaboostPredict(vector, test_emo)
	knearestNeighboursPredict(vector, test_emo)






