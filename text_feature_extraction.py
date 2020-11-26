import nltk
from nltk.corpus import stopwords
from nltk.sentiment.util import *
from nltk.util import ngrams
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

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

if __name__ == "__main__":
	train_df = pd.read_csv("text_data/train_sent_emo.csv")
	train_utterance = train_df["Utterance"].apply(preprocess)
	train_emo = train_df["Emotion"]

	dev_df = pd.read_csv("text_data/dev_sent_emo.csv")
	dev_utterance = dev_df["Utterance"].apply(preprocess)
	dev_emo = dev_df["Emotion"]

	test_df = pd.read_csv("text_data/test_sent_emo.csv")
	test_utterance = test_df["Utterance"].apply(preprocess)
	test_emo = test_df["Emotion"]
