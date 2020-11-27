import nltk
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.util import ngrams
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

import numpy as np
import pandas as pd

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
	print("hello")

	analysis = nltk.FreqDist(tokensCombined)
	del tokensCombined
	frequencyDict = dict([(m, n) for m, n in analysis.items() if n > 5])
	lenfrequencyDict = len(frequencyDict)
	wordIndex = {}
	for i, key in enumerate(frequencyDict.keys()):
		wordIndex[key] = i
	frequencyDict.clear()
	print(lenfrequencyDict)

	# f = open("dict/unigram_dict.pkl", "wb")
	# pickle.dump(wordIndex, f)

	unigramVector = np.zeros([n, lenfrequencyDict], dtype=np.bool_)
	for i, tokens in enumerate(utterance_tokenized):
		arr = np.zeros([lenfrequencyDict])
		for token in tokens:
			if token in wordIndex:
				arr[wordIndex[token]] = 1
		unigramVector[i] = arr 
	return unigramVector

def bigram(utterance, utterance_tokenized):
	print("hello2")
	n = len(utterance)
	bigraminUtterance= []
	allbigrams = []
	for tokenized in utterance_tokenized:
		bigrams = [' '.join(grams) for grams in ngrams(tokenized, 2)]
		bigraminUtterance.append(bigrams)
		allbigrams.extend(bigrams)

	print("hello2")
	analysis = nltk.FreqDist(allbigrams)
	del allbigrams
	frequencybigramDict = dict([(m, n) for m, n in analysis.items() if n > 10])
	lenfrequencybigramDict = len(frequencybigramDict)
	print(lenfrequencybigramDict)
	bigramIndexDict = {}
	print("hello2")

	for i, key in enumerate(frequencybigramDict.keys()):
		bigramIndexDict[key] = i
	frequencybigramDict.clear()

	# f = open("dict/bigram_dict.pkl", "wb")
	# pickle.dump(bigramIndexDict, f)
	# print("hello2")

	bigramVector = np.zeros([n, lenfrequencybigramDict], dtype=np.bool_)
	for i, bigramUtterance in enumerate(bigraminUtterance):
		arr = np.zeros([lenfrequencybigramDict])
		for bigram in bigramUtterance:
			if bigram in bigramIndexDict:
				arr[bigramIndexDict[bigram]] = 1
		bigramVector[i] = arr

	del bigraminUtterance



if __name__ == "__main__":
	train_utterance_tokenized = []
	train_df = pd.read_csv("text_data/train_sent_emo.csv")
	# train_utterance = train_df["Utterance"].apply(preprocess).values.tolist()
	train_utterance = train_df["Utterance"].values.tolist()
	train_emo = train_df["Emotion"]
	tokenized(train_utterance, train_utterance_tokenized)

	# dev_utterance_tokenized = []
	# dev_df = pd.read_csv("text_data/dev_sent_emo.csv")
	# dev_utterance = dev_df["Utterance"].apply(preprocess).values.tolist()
	# dev_utterance = dev_df["Utterance"].values.tolist()
	# dev_emo = dev_df["Emotion"]	
	# tokenized(dev_utterance, dev_utterance_tokenized)

	# test_utterance_tokenized = []
	# test_df = pd.read_csv("text_data/test_sent_emo.csv")
	# test_utterance = test_df["Utterance"].apply(preprocess).values.tolist()
	# test_utterance = test_df["Utterance"].values.tolist()
	# test_emo = test_df["Emotion"]
	# tokenized(test_utterance, test_utterance_tokenized)



	unigramVector_train = unigram(train_utterance, train_utterance_tokenized)
	bigramVector_train = bigram(train_utterance, train_utterance_tokenized)



