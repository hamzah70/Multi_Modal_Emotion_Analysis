import opensmile
import time
import os
import os.path
import numpy as np
import pandas as pd
import pickle
s = time.time()

def dictToarr():
    train_utterance_tokenized = []
    train_df = pd.read_csv("text_data/train_sent_emo.csv")

    train_dialogue = train_df["Dialogue_ID"].values.tolist()
    train_utterance = train_df["Utterance_ID"].values.tolist()

    f = open("audio_features_dict.p", "rb")
    d = pickle.load(f)

    audioFeature = np.zeros([len(train_dialogue), 528])
    for i in range(len(train_dialogue)):
        dialogueID = train_dialogue[i]
        utteranceID = train_utterance[i]
        fname = "dia" + str(dialogueID) + "_utt" + str(utteranceID) + ".mp4"
        try:
            audioFeature[i] = np.nan_to_num(d[fname])
        except:
            audioFeature[i]=np.zeros([528])

    return audioFeature

def audioFeature():
    s = time.time()
    smile2 = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01b,
                             feature_level=opensmile.FeatureLevel.Functionals, num_channels=2)
    smile6 = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv01b,
                             feature_level=opensmile.FeatureLevel.Functionals, num_channels=6)

    directory_in_str = "data/MELD.Raw/train/train_splits"

    vec_528 = smile6.process_file(directory_in_str+"/dia47_utt11.mp4")
    vec_176 = smile2.process_file(directory_in_str+"/dia645_utt11.mp4")
    col_list = (vec_176.append([vec_528])).columns.tolist()


    X_dict = {}
    directory = os.fsencode(directory_in_str)
    i = 0
    files = sorted(os.listdir(directory))
    for file in sorted(os.listdir(directory), key=lambda s: s.lower()):
        filename = os.fsdecode(file)

        try:
            feature_vec = smile6.process_file(directory_in_str + "/"+filename)
            X_dict[filename] = feature_vec
        except RuntimeError:
            feature_vec = smile2.process_file(directory_in_str + "/"+filename)
            feature_vec = feature_vec.reindex(columns=col_list, fill_value=0)
            X_dict[filename] = feature_vec
        except:
            continue

        i += 1
        if i%100==0:
            print(i)

    pickle.dump(X_dict, open('audio_features.p', 'wb'))

    print(time.time()-s)

if __name__ == '__main__':
    if not os.path.isfile('audio_features_dict.p'):
        print("hello")
        audioFeature()
    dictToarr()

