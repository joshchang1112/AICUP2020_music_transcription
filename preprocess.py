import numpy as np
import os
import json
import pickle
import copy
from dataset import MyData
from sklearn import preprocessing

def preprocess(data_seq, label):
    new_label= []
    for i in range(len(label)):
        label_of_one_song= []
        cur_note= 0
        cur_note_onset= label[i][cur_note][0]
        cur_note_offset= label[i][cur_note][1]
        cur_note_pitch= 2**((label[i][cur_note][2]-69)/12) * 440

        for j in range(len(data_seq[i])):
            cur_time= j* 0.032+ 0.016
        
            if abs(cur_time - cur_note_onset) < 0.017:
                label_of_one_song.append(np.array([1, 0, cur_note_pitch]))

            elif cur_time < cur_note_onset or cur_note >= len(label[i]):
                label_of_one_song.append(np.array([0, 0, 0]))

            elif abs(cur_time - cur_note_offset) < 0.017:
                label_of_one_song.append(np.array([0, 1, cur_note_pitch]))
                cur_note= cur_note+ 1
                if cur_note < len(label[i]):
                    cur_note_onset = label[i][cur_note][0]
                    cur_note_offset = label[i][cur_note][1]
                    cur_note_pitch = 2**((label[i][cur_note][2]-69)/12) * 440
            else:
                label_of_one_song.append(np.array([0, 0, cur_note_pitch]))
        new_label.append(label_of_one_song)

    return new_label

# Traib data
THE_FOLDER = "./MIR-ST500"
#THE_FOLDER = "./AIcup_testset_ok"
data_seq= []
frame_num = []
label= []
a = os.listdir(THE_FOLDER)
a.sort(key= lambda x: int(x))
for the_dir in a:
    print (the_dir)
    if not os.path.isdir(THE_FOLDER + "/" + the_dir):
        continue


    json_path = THE_FOLDER + "/" + the_dir+ f"/{the_dir}_feature.json"
    gt_path= THE_FOLDER+ "/" +the_dir+ "/"+ the_dir+ "_groundtruth.txt"

    youtube_link_path= THE_FOLDER+ "/" + the_dir+ "/"+ the_dir+ "_link.txt"

    with open(json_path, 'r') as json_file:
        temp = json.loads(json_file.read())

    gtdata = np.loadtxt(gt_path)

    data = []
    for key, value in temp.items():
        data.append(value)

    data = np.array(data).T
    #print(data.shape)
    #data = preprocessing.scale(data)
    data_seq.extend(data)
    frame_num.append(len(data))
    label.append(gtdata)

# np_data = preprocessing.scale(data_seq)
# data_seq = []
# start = 0
# for idx in frame_num:
#     data_seq.append(np_data[start:start+idx, :])
#     start+=idx

#print(np_data.shape)

# Preprocess test data only
# scaler = preprocessing.StandardScaler()
# scaler.fit(data_seq)
# print(scaler.mean_)

groundtruth = copy.deepcopy(label)

for i in range(len(groundtruth)):
    for j in range(len(groundtruth[i])):
        groundtruth[i][j][2] = 2**((groundtruth[i][j][2]-69)/12) * 440


label= preprocess(data_seq, label)
train_data = MyData(data_seq, label, groundtruth)
#test_data = MyData(data_seq)
with open("feature_pickle.pkl", 'wb') as pkl_file:
#with open("test.pkl", 'wb') as pkl_file:
    pickle.dump(train_data, pkl_file)

# Preprocess Test data
# THE_FOLDER = "./AIcup_testset_ok"
# data_seq= []
# frame_num = []
# label= []
# a = os.listdir(THE_FOLDER)
# a.sort(key= lambda x: int(x))
# for the_dir in a:
#     print (the_dir)
#     if not os.path.isdir(THE_FOLDER + "/" + the_dir):
#         continue

#     json_path = THE_FOLDER + "/" + the_dir+ f"/{the_dir}_feature.json"

#     youtube_link_path= THE_FOLDER+ "/" + the_dir+ "/"+ the_dir+ "_link.txt"

#     with open(json_path, 'r') as json_file:
#         temp = json.loads(json_file.read())

#     data = []
#     for key, value in temp.items():
#         data.append(value)

#     data = np.array(data).T
#     #print(data.shape)
#     #data = preprocessing.scale(data)
#     data_seq.extend(data)
#     frame_num.append(len(data))

# data = scaler.transform(data_seq)
# data_seq = []
# start = 0
# for idx in frame_num:
#     data_seq.append(data[start:start+idx, :])
#     start+=idx

# test_data = MyData(data_seq)
# with open("test.pkl", 'wb') as pkl_file:
#     pickle.dump(test_data, pkl_file)
