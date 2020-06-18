import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from utils import set_seed
from model import LSTM_set, LSTM_pitch
from dataset import MyData
from train import do_training
from test import testing
import pickle

# DATA_FOLDER = "./MIR-ST500"
# data_seq= []
# label= []

# train_data= None
'''
# train
with open("feature_pickle.pkl", 'rb') as pkl_file:
    train_data= pickle.load(pkl_file)


BATCH_SIZE = 8
train_num = 400
train_size = train_num
test_size = 500 - train_size

input_dim = 23
hidden_size = 128

set_seed(208)
train_data, val_data = random_split(train_data, [train_size, test_size])

train_loader = DataLoader(dataset=train_data, batch_size= BATCH_SIZE, shuffle=True, 
    collate_fn=MyData.collate_fn)
val_loader = DataLoader(dataset=val_data, batch_size= BATCH_SIZE, shuffle=False, 
    collate_fn=MyData.collate_fn)


model = LSTM_set(input_dim, hidden_size)

# model_1.load_state_dict(torch.load("./model_1ST_45.pt"))
# model_2.load_state_dict(torch.load("./model_2ST_45.pt"))

device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'

model = model.to(device)
print("use",device,"now!")

model = do_training(model, train_loader, val_loader, device)
'''
# test
with open("test.pkl", 'rb') as pkl_file:
    test_data= pickle.load(pkl_file)

input_dim = 23
hidden_size = 128
BATCH_SIZE = 8

set_seed(0)

test_loader = DataLoader(dataset=test_data, batch_size= BATCH_SIZE, shuffle=False, 
      collate_fn=MyData.collate_fn)

model = LSTM_set(input_dim, hidden_size)

model.load_state_dict(torch.load("./set_model.pkl"))

device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'

model = model.to(device)
print("use",device,"now!")


output = testing(model, test_loader, device)

output_json = {}
for i in range(1, 1501):
    output_json[str(i)] = output[i-1]

import json
with open('answer.json', 'w') as f_obj:
    json.dump(output_json, f_obj)
