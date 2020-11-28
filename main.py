import opensmile
import time
import os
import numpy
import pickle
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
    # if i == 10:
    #     break

pickle.dump(X_dict, open('audio_features.p', 'wb'))

# a_f=pickle.load(open('audio_features.p','rb'))
# print(a_f)


print(time.time()-s)
