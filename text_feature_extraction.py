import time

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.util import ngrams
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler

from lexiconFeatureVector import lexicons
from model_train import *
import audio_feature_extraction

from sent2vec.vectorizer import Vectorizer

cachedStopWords = set(stopwords.words("english"))

def preprocess(sent):
	sent = word_tokenize(sent)
	lemmatizer = WordNetLemmatizer()
	ps = PorterStemmer()
	sent = [words for words in sent if words not in cachedStopWords]
	sent = [lemmatizer.lemmatize(words) for words in sent]
	sent = " ".join([ps.stem(words) for words in sent])
	return sent

def tokenized(utterance, tokenizedarr):
	for sent in utterance:
		tokenizedarr.append(word_tokenize(sent))

def unigram(utterance, utterance_tokenized):
	n = len(utterance)
	tokensCombined = []

	for i, tokens in enumerate(utterance_tokenized):
		tokensCombined.extend(tokens)
	# print("hello")

	analysis = nltk.FreqDist(tokensCombined)
	del tokensCombined
	frequencyDict = dict([(m, n) for m, n in analysis.items() if n > 5])
	lenfrequencyDict = len(frequencyDict)
	wordIndex = {}
	for i, key in enumerate(frequencyDict.keys()):
		wordIndex[key] = i
	frequencyDict.clear()
	print(lenfrequencyDict)

	f = open("dict/unigram_dict.pkl", "wb")
	pickle.dump(wordIndex, f)

	unigramVector = np.zeros([n, lenfrequencyDict], dtype=np.bool_)
	for i, tokens in enumerate(utterance_tokenized):
		arr = np.zeros([lenfrequencyDict])
		for token in tokens:
			if token in wordIndex:
				arr[wordIndex[token]] = 1
		unigramVector[i] = arr 
	return unigramVector

def bigram(utterance, utterance_tokenized):
	# print("hello2")
	n = len(utterance)
	bigraminUtterance= []
	allbigrams = []
	for tokenized in utterance_tokenized:
		bigrams = [' '.join(grams) for grams in ngrams(tokenized, 2)]
		bigraminUtterance.append(bigrams)
		allbigrams.extend(bigrams)

	# print("hello2")
	analysis = nltk.FreqDist(allbigrams)
	del allbigrams
	frequencybigramDict = dict([(m, n) for m, n in analysis.items() if n > 10])
	lenfrequencybigramDict = len(frequencybigramDict)
	# print(lenfrequencybigramDict)
	bigramIndexDict = {}
	# print("hello2")

	for i, key in enumerate(frequencybigramDict.keys()):
		bigramIndexDict[key] = i
	frequencybigramDict.clear()

	f = open("dict/bigram_dict.pkl", "wb")
	pickle.dump(bigramIndexDict, f)
	# print("hello2")

	bigramVector = np.zeros([n, lenfrequencybigramDict], dtype=np.bool_)
	for i, bigramUtterance in enumerate(bigraminUtterance):
		arr = np.zeros([lenfrequencybigramDict])
		for bigram in bigramUtterance:
			if bigram in bigramIndexDict:
				arr[bigramIndexDict[bigram]] = 1
		bigramVector[i] = arr

	del bigraminUtterance
	return bigramVector

def sent2vec_feature(utterances):
	vectorizer = Vectorizer()
	vectorizer.bert(utterances)
	return vectorizer.vectors

def extend(arr1, arr2):
	for i in range(len(arr2)):
		arr1 = np.append(arr1, arr2[i], 0)
		

if __name__ == "__main__":
	train_utterance_tokenized = []
	train_df = pd.read_csv("text_data/train_sent_emo.csv")
	# train_utterance = train_df["Utterance"].apply(preprocess).values.tolist()
	train_utterance = train_df["Utterance"].values.tolist()
	train_emo = train_df["Emotion"].values.tolist()
	tokenized(train_utterance, train_utterance_tokenized)

	dev_utterance_tokenized = []
	dev_df = pd.read_csv("text_data/dev_sent_emo.csv")
	# dev_utterance = dev_df["Utterance"].apply(preprocess).values.tolist()
	dev_utterance = dev_df["Utterance"].values.tolist()
	dev_emo = dev_df["Emotion"].values.tolist()
	tokenized(dev_utterance, dev_utterance_tokenized)

	# test_utterance_tokenized = []
	# test_df = pd.read_csv("text_data/test_sent_emo.csv")
	# test_utterance = test_df["Utterance"].apply(preprocess).values.tolist()
	# test_utterance = test_df["Utterance"].values.tolist()
	# test_emo = test_df["Emotion"].values.tolist()
	# tokenized(test_utterance, test_utterance_tokenized)

	# print(len(set(train_emo)))
	# print(set(train_emo))
	# print(train_emo.count("neutral"))

	train_utterance.extend(dev_utterance)
	train_emo.extend(dev_emo)
	train_utterance_tokenized.extend(dev_utterance_tokenized)


	unigramVector_train = unigram(train_utterance, train_utterance_tokenized)
	bigramVector_train = bigram(train_utterance, train_utterance_tokenized)
	lexicon_train = lexicons(train_utterance_tokenized)
	# sent2vec_train = sent2vec_feature(train_utterance)
	audio_train = audio_feature_extraction.dictToarr("train")
	audio_dev = audio_feature_extraction.dictToarr("dev")

	l1 = audio_train.tolist()
	l2 = audio_dev.tolist()
	l1.extend(l2)

	audio_train = np.array(l1)

	# unigramVector_dev = unigram(dev_utterance, dev_utterance_tokenized)
	# bigramVector_dev = bigram(dev_utterance, dev_utterance_tokenized)
	# lexicon_dev = lexicons(dev_utterance_tokenized)
	# # sent2vec_train = sent2vec_feature(train_utterance)
	# audio_dev = audio_feature_extraction.dictToarr("dev")

	# extend(unigramVector_train, unigramVector_dev)
	# extend(bigramVector_train, bigramVector_dev)
	# extend(lexicon_train, lexicon_dev)
	# extend(audio_train, audio_dev)



	f = open("features/unigramVector_train.pkl", "wb")
	f = open("features/bigramVector_train.pkl", "wb")
	f = open("features/lexicon_train.pkl", "wb")
	f = open("features/audio_train.pkl", "wb")

	pickle.dump(unigramVector_train, f)
	pickle.dump(bigramVector_train, f)
	pickle.dump(lexicon_train, f)
	pickle.dump(audio_train, f)

	vector = np.zeros([len(train_utterance), len(unigramVector_train[0]) + len(bigramVector_train[0]) + len(lexicon_train[0]) + len(audio_train[0])])
	for i in range(len(train_utterance)):
		vector[i] = np.concatenate((unigramVector_train[i], bigramVector_train[i], lexicon_train[i], audio_train[i]))

	sel = SelectKBest(f_classif, k=500)
	vector = sel.fit_transform(vector, train_emo)

	sc = StandardScaler()
	vector = sc.fit_transform(vector)

	start = time.time()
	print("Before PCA: ", vector.shape)
	pca = PCA(n_components=0.95)
	pca.fit(vector)
	vector = pca.transform(vector)
	end = time.time()
	print("time taken to pca: ", end-start, "  ", vector.shape)


	label_encoder = LabelEncoder()
	label_encoder.fit(train_emo)
	train_emo = label_encoder.transform(train_emo)

	f1 = open("models/kbest.pkl", "wb")
	f2 = open("models/scaler.pkl", "wb")
	f3 = open("models/pca.pkl", "wb")
	f4 = open("models/labelencoder.pkl", "wb")

	pickle.dump(sel, f1)
	pickle.dump(sc, f2)
	pickle.dump(pca, f3)
	pickle.dump(label_encoder, f4)

	svmModel(vector, train_emo)
	randomForestModel(vector, train_emo)
	mlpModel(vector, train_emo)
	# cnn(vector,train_emo)





