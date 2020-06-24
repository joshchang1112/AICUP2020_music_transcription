import numpy as np
import os
import json
import pickle
import copy
from dataset import MyData
from sklearn import preprocessing

def preprocess(data_seq, label, _range):
    new_label= []
    for i in range(len(label)):
        label_of_one_song= []
        cur_note= 0
        cur_note_onset= label[i][cur_note][0]
        cur_note_offset= label[i][cur_note][1]
        #cur_note_pitch= 2**((label[i][cur_note][2]-69)/12) * 440
        tmp = 0
        for j in range(len(data_seq[i])):
            cur_time= (j+_range[i][0])* 0.032+ 0.016
        
            if abs(cur_time - cur_note_onset) < 0.017:
                tmp = 1
                label_of_one_song.append(np.array([1, 0]))

            elif cur_time < cur_note_onset or cur_note >= len(label[i]):
                label_of_one_song.append(np.array([0, 0]))

            elif abs(cur_time - cur_note_offset) < 0.017:
                tmp = 2
                label_of_one_song.append(np.array([0, 1]))
                cur_note= cur_note+ 1
                if cur_note < len(label[i]):
                    cur_note_onset = label[i][cur_note][0]
                    cur_note_offset = label[i][cur_note][1]
                    #cur_note_pitch = 2**((label[i][cur_note][2]-69)/12) * 440
            else:
                if cur_note_onset < cur_time and tmp == 2:
                    label_of_one_song.append(np.array([1, 0]))
                    tmp = 1
                else:
                    label_of_one_song.append(np.array([0, 0]))
        new_label.append(label_of_one_song)

    return new_label

# Preprocess Train data
THE_FOLDER = "./MIR-ST500"
#THE_FOLDER = "./AIcup_testset_ok"
data_seq= []
frame_num = []
label= []
vocal_pitch = []
vocal_pitch_2 = []
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
    zcr_2 = np.expand_dims(data[:, 0] ** 2, axis=1)
    v_2 = np.expand_dims(data[:, 22] ** 2, axis=1)
    e_2 = np.expand_dims(data[:, 1] ** 2, axis=1)
    ep_2 = np.expand_dims(data[:, 2] ** 2, axis=1)
    sp_2 = np.expand_dims(data[:, 3] ** 2, axis=1)

    data = np.concatenate((data, zcr_2, v_2, e_2, ep_2, sp_2), axis=1)
    #print(data.shape)
    #data = preprocessing.scale(data)
    data_seq.extend(data)
    frame_num.append(len(data))
    label.append(gtdata)
    vocal_pitch.append(data[:, 22])
    vocal_pitch_2.extend(data[:, 22])


_range = []
for i in range(len(vocal_pitch)):
    tmp = 0
    start = 0
    end = 0
    for j in range(len(vocal_pitch[i])):
        if vocal_pitch[i][j] != 0:
            if tmp == 0:
                start = j-5
            tmp = 1
            
        elif vocal_pitch[i][j] == 0 and tmp == 1:
            end = j+5
            tmp = 2
    _range.append([start, end])


#train data standardlization
scaler = preprocessing.StandardScaler()
scaler.fit(data_seq)

np_data = preprocessing.scale(data_seq)
data_seq = []
vocal_pitch_3 = []
now = 0
for idx, (start, end) in zip(frame_num, _range):
    data_seq.append(np_data[now+start:now+end, :])
    vocal_pitch_3.append(vocal_pitch_2[now+start:now+end])
    now+=idx

groundtruth = copy.deepcopy(label)

for i in range(len(groundtruth)):
    for j in range(len(groundtruth[i])):
        groundtruth[i][j][2] = 2**((groundtruth[i][j][2]-69)/12) * 440


# label= preprocess(data_seq, label, _range)
# train_data = MyData(data_seq,  _range, label, groundtruth, vocal_pitch_3)
# #test_data = MyData(data_seq)
# with open("feature_pickle.pkl", 'wb') as pkl_file:
#    pickle.dump(train_data, pkl_file)


# #Preprocess Test data
THE_FOLDER = "./AIcup_testset_ok"
data_seq= []
frame_num = []
label= []
vocal_pitch = []
vocal_pitch_2 = []
a = os.listdir(THE_FOLDER)
a.sort(key= lambda x: int(x))
for the_dir in a:
    print (the_dir)
    if not os.path.isdir(THE_FOLDER + "/" + the_dir):
        continue

    json_path = THE_FOLDER + "/" + the_dir+ f"/{the_dir}_feature.json"

    youtube_link_path= THE_FOLDER+ "/" + the_dir+ "/"+ the_dir+ "_link.txt"

    with open(json_path, 'r') as json_file:
        temp = json.loads(json_file.read())

    data = []
    for key, value in temp.items():
        data.append(value)

    data = np.array(data).T
    zcr_2 = np.expand_dims(data[:, 0] ** 2, axis=1)
    v_2 = np.expand_dims(data[:, 22] ** 2, axis=1)
    e_2 = np.expand_dims(data[:, 1] ** 2, axis=1)
    ep_2 = np.expand_dims(data[:, 2] ** 2, axis=1)
    sp_2 = np.expand_dims(data[:, 3] ** 2, axis=1)

    data = np.concatenate((data, zcr_2, v_2, e_2, ep_2, sp_2), axis=1)
    #print(data.shape)
    #data = preprocessing.scale(data)
    data_seq.extend(data)
    frame_num.append(len(data))
    vocal_pitch.append(data[:, 22])
    vocal_pitch_2.extend(data[:, 22])


data_2 = scaler.transform(data_seq)

_range = []
for i in range(len(vocal_pitch)):
    tmp = 0
    start = 0
    end = 0
    for j in range(len(vocal_pitch[i])):
        if vocal_pitch[i][j] != 0:
            if tmp == 0:
                start = j-5
            tmp = 1
            
        elif vocal_pitch[i][j] == 0 and tmp == 1:
            end = j+5
            tmp = 2
    _range.append([start, end])

data_seq = []
vocal_pitch_3 = []
now = 0
for idx, (start, end) in zip(frame_num, _range):
    data_seq.append(data_2[now+start:now+end, :])
    vocal_pitch_3.append(vocal_pitch_2[now+start:now+end])
    now+=idx

test_data = MyData(data_seq, _range, vocal_pitch=vocal_pitch_3)
with open("test.pkl", 'wb') as pkl_file:
    pickle.dump(test_data, pkl_file)

