from tqdm import tqdm
import math
import time
import numpy as np
def bingliu_mpqa(utterance_tokenized, file):
    feat_ = []
    dict1_bing = {}
    for line in file:
        x = line.split("\t")
        dict1_bing[x[0] + "_" + x[1][:-1]] = 1
    i=0
    for tokens in utterance_tokenized:
        res = np.array([0,0,0,0])
        for token in tokens:
            pos = (token + "_positive")
            neg = (token + "_negative")
            if (pos in dict1_bing):
                res[0]+=1
                res[1]+=1
            elif (neg in dict1_bing):
                res[1]-=1
        if res[0]>0:
            res[2]=1
        if tokens!=[]:
            pos = tokens[-1] + "_positive"
            neg = tokens[-1] + "_negative"
            if pos in dict1_bing:
                res[3]=1
            elif neg in dict1_bing:
                res[3]=-1
        feat_.append(res)
    return np.array(feat_)

def SENT140(X):
    #sentiment140
    dict1_S140 = {}
    with open("lexicons/3. Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt", 'r') as fd:
        for line in fd:
            x = line.split("	")
            dict1_S140[x[0]] = float(x[1])
    
    feat_ = []
    for tokens in X:
        sent140 = [0,0,0,0]
        cnt = 0
        for token in tokens:
            if("#" not in token):
                cnt += 1
                if(token in dict1_S140):
                    sent140[0] += (dict1_S140[token] > 0)
                    sent140[1] += dict1_S140[token]
                    sent140[2] = max(sent140[2],dict1_S140[token])
        if(len(tokens) >= 1 and tokens[-1] in dict1_S140):
        	sent140[3] = (dict1_S140[tokens[-1]] > 0)
        feat_.append(sent140)
    return np.array(feat_)
# print()
def NRC_EMOTION(X):
    #NRC emotion
    dict1_NRC = {}
    cnt_r = 0
    len1 = 0;
    with open("lexicons/6. NRC-10-expanded.csv", 'r') as fd:
        for line in fd:
            if(cnt_r == 0):
                cnt_r += 1
                continue;
            x = line.split("	")
            dict1_NRC[x[0]] = [float(i) for i in x[1:]]
            len1 = len(x[1:])
    feat_ = []
    for e,tokens in tqdm(enumerate(X)):
        emo_score = [[0,0,0,0] for i in range(len1)]
        cnt = 0
        for token in tokens:
            if("#" in token):
                continue
            cnt += 1
            if(token in dict1_NRC):
                for i,val in enumerate(dict1_NRC[token]):
                	emo_score[i][0] += (val > 0)
                	emo_score[i][1] += val
                	emo_score[i][2] = max(emo_score[i][2],val)
        if(len(tokens) >= 1 and tokens[-1] in dict1_NRC):
        	for i,val in enumerate(dict1_NRC[token]):
        		emo_score[i][3] = (val > 0)
       	res = []
       	for i in emo_score:
       		res.extend(i)
       	feat_.append(res)
    return np.array(feat_)
# print()
def NRC_HASHTAG_SENT(X):
    #NRC hashtag
    dict1_NRC = {}
    with open("lexicons/7. NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt", 'r') as fd:
        for line in fd:
            x = line.split("	")
            dict1_NRC[x[0]] = float(x[1])
    feat_ = []
    for tokens in X:
        cnt = 0
        f = [0,0,0,0]
        for token in tokens:
            if("#" not in token):
                continue
            cnt += 1
            if(token in dict1_NRC):
            	f[0] += (dict1_NRC[token] > 0)
            	f[1] += dict1_NRC[token]
            	f[2] = max(f[2],dict1_NRC[token])
       	if(len(tokens) >= 1 and tokens[-1] in dict1_NRC):
       		f[3] = (dict1_NRC[tokens[-1]] > 0)
        feat_.append(f)
    return np.array(feat_)

def lexicons(utterance_tokenized):
    filebingliu = open("lexicons/1. BingLiu.csv", "r")
    filempqa = open("lexicons/2. mpqa.txt", "r")

    start = time.time()
    bingliu = bingliu_mpqa(utterance_tokenized, filebingliu)
    print("bing liu complete")
    mpqa = bingliu_mpqa(utterance_tokenized, filempqa)
    print("mpqa complete")
    sent140 = SENT140(utterance_tokenized)
    print("sent 140 complete")
    nrcemotion = NRC_EMOTION(utterance_tokenized)
    print("nrc emotion complete")
    nrchashtag = NRC_HASHTAG_SENT(utterance_tokenized)
    print("nrc hashtag complete")
    end = time.time()
    print("time to calculate lexicons: ", end-start)

    # y = len(bingliu[0]) + len([mpqa[0]]) + len(sent140[0]) + len(nrcemotion[0]) + len(nrchashtag[0])
    feature = np.zeros([len(utterance_tokenized), 56])
    for i in range(len(utterance_tokenized)):
        feature[i] = np.concatenate((bingliu[i], mpqa[i], sent140[i], nrcemotion[i], nrchashtag[i]))
    return feature

if __name__=='__main__':
    lexicons(utterance_tokenized)




