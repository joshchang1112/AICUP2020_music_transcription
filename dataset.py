import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class MyData(Dataset):
    def __init__(self, data_seq, label=None, groundtruth=None, vocal_pitch=None):
        self.data_seq = data_seq
        self.label= label
        self.groundtruth = groundtruth
        self.vocal_pitch = vocal_pitch

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        if self.label == None:
            return {
                'data': self.data_seq[idx],
                'vocal_pitch': self.vocal_pitch[idx]
            }
        return {
            'data': self.data_seq[idx],
            'label': self.label[idx],
            'groundtruth': self.groundtruth[idx],
            'vocal_pitch': self.vocal_pitch[idx]
        }

    def collate_fn(samples):
        batch = {}
        #print (samples[0]['data'].shape)
        batch['data_lens'] = [len(sample['data']) for sample in samples]
        temp= [torch.from_numpy(np.array(sample['data'], dtype= np.float32)) for sample in samples]
        padded_data = rnn_utils.pad_sequence(temp, batch_first=True, padding_value= 0)
        batch['data']= torch.Tensor(padded_data)
        padded_len = max(batch['data_lens'])
        #print(padded_len)
        # need to comment out when predicting
        #batch['label']= torch.Tensor([pad_to_len(sample['label'], padded_len) for sample in samples])
        #batch['groundtruth']= [np.array(sample['groundtruth'], dtype= np.float32) for sample in samples]
        batch['vocal_pitch']= [np.array(sample['vocal_pitch'], dtype= np.float32) for sample in samples]


        return batch

def pad_to_len(arr, padded_len):

    #arr = np.array(arr, dtype= np.float32)
    new_arr = arr.copy()
    length_arr = len(arr)
    if length_arr < padded_len: 
        for i in range(padded_len - len(arr)):
            new_arr.append([0, 0])

    return np.array(new_arr, dtype= np.float32)
