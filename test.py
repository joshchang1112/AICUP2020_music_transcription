import json
import os
import numpy as np
import pickle
import sys
import time
import mir_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
from tensorboardX import SummaryWriter 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import copy
from utils import post_processing

def testing(net, test_loader, device):
    net_1 = net[0]
    net_2 = net[1]
    net_1.eval()
    net_2.eval()
    predict = []
    for idx, sample in enumerate(test_loader):
        
        data = torch.Tensor(sample['data'])
        data_lens = sample['data_lens']
        data_length= list(data.shape)[0]

        data = data.to(device, dtype=torch.float)
        output1 = net_1(data)
        output2 = net_2(data)
        answer = post_processing(output1, output2)
        predict.extend(answer)
        print(len(predict))
    
    return predict
