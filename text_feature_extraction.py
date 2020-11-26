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

def unigram(utterance):
	n = len(utterance)
	tokensCombined = []
	tokenizedUtterances = []

	for u in utterance:
		tokenizedUtterances.append(word_tokenize(u))

	for i, tokens in enumerate(tokenizedUtterances):
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
	for i, tokens in enumerate(tokenizedUtterances):
		arr = np.zeros([lenfrequencyDict])
		for token in tokens:
			if token in wordIndex:
				arr[wordIndex[token]] = 1
		unigramVector[i] = arr
	return unigramVector

if __name__ == "__main__":
	train_df = pd.read_csv("text_data/train_sent_emo.csv")
	# train_utterance = train_df["Utterance"].apply(preprocess)
	train_utterance = train_df["Utterance"]
	train_emo = train_df["Emotion"]

	# dev_df = pd.read_csv("text_data/dev_sent_emo.csv")
	# dev_utterance = dev_df["Utterance"].apply(preprocess)
	# dev_emo = dev_df["Emotion"]

	# test_df = pd.read_csv("text_data/test_sent_emo.csv")
	# test_utterance = test_df["Utterance"].apply(preprocess)
	# test_emo = test_df["Emotion"]

	unigramVector_train = unigram(train_utterance.values.tolist())



