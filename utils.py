import os
import torch
import torch.nn as nn
import random
import numpy as np

def post_processing(output, vocal_pitch):

    # print (torch.mean(output1))
    threshold = 0.01

    notes= []
    for i in range(len(output)):
        this_onset= None
        this_offset= None
        this_pitch= None
        notes.append([])
        for j in range(len(output[i])):
            
            if j >= len(vocal_pitch[i])-2:
                break
            if vocal_pitch[i][j] == 0 and vocal_pitch[i][j+1] == 0 and vocal_pitch[i][j+2] == 0:
                continue

            if output[i][j][0] >= threshold and this_onset == None:
                this_onset= j
                if j == len(output[i]) - 1:
                    continue
                if output[i][j+1][0] >= threshold and output[i][j+1][0]>output[i][j][0]:
                    this_onset=None
            
            elif output[i][j][1] >= threshold and this_onset != None and this_onset+ 1 < j and this_offset == None:
                this_offset= j
                if j == len(output[i]) - 1:
                    pass
                elif output[i][j+1][1] >= threshold and output[i][j+1][1]>output[i][j][1]:
                    this_offset=None
                    continue
                this_pitch= int(round(np.median(vocal_pitch[i][this_onset:this_offset+1])))
                notes[i].append([this_onset* 0.032+ 0.016, this_offset* 0.032+ 0.016, this_pitch])

                this_onset= None
                this_offset= None
                this_pitch= None

    # print (np.array(notes))
    return notes

def set_seed(SEED):
    SEED = 0
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
