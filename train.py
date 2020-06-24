import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import mir_eval
import time
from utils import post_processing

#def do_training(net, train_loader, val_loader, device):
def do_training(on_net, off_net, train_loader, val_loader, device):
    num_epoch = 60
    criterion_set = nn.BCELoss()
    #criterion_pitch = nn.SmoothL1Loss()
    #criterion_pitch = nn.CrossEntropyLoss()
    train_loss= 0.0
    total_length= 0
    best_f1 = 0
    best_val_loss = 100

    optimizer = optim.AdamW(list(on_net.parameters()) +list(off_net.parameters()), lr= 2e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=4)


    for epoch in range(num_epoch):
        total_length= 0.0
        start_time = time.time()
        train_loss= 0.0
        total_onset_loss = 0.0
        total_offset_loss=0.0

        COn = 0
        COnP = 0
        COnPOff = 0
        weighted_f1 = 0
        count = 0 
        for param_group in optimizer.param_groups:
             print("lr: {}".format(param_group['lr']))

        on_net.train()
        off_net.train()
        for batch_idx, sample in enumerate(train_loader):

            # print(len(sample['label']))
            # print(sample['label'][0].shape)
            # print(sample['label'][1].shape)
            data = sample['data']
            target = sample['label']
            data_lens = sample['data_lens']
            vocal_pitch = sample['vocal_pitch']
            _range = sample['range']

            data_length= list(data.shape)[0]

            data = data.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            optimizer.zero_grad()
            on_output = on_net(data)
            off_output = off_net(data)

            on_loss = criterion_set(on_output, torch.narrow(target, dim= 2, start= 0, length= 1))
            off_loss = criterion_set(off_output, torch.narrow(target, dim= 2, start= 1, length= 1))
            #loss = criterion_set(output, torch.narrow(target, dim= 2, start= 0, length= 2))
            loss = on_loss + off_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total_length = total_length + 1
            output = torch.cat([on_output, off_output], dim=2)
            if epoch >= 10:
                gt = np.array(sample['groundtruth'])
                predict = post_processing(output, vocal_pitch, _range)
                #predict = np.array(predict)          

                for i in range(len(predict)):
                    count += 1
                    tmp_predict = np.where(np.array(predict[i]) > 0, np.array(predict[i]), 0.0001)
                    if tmp_predict.shape[0] == 0:
                        continue
                    score = mir_eval.transcription.evaluate(gt[i][:, :2], gt[i][:, 2], tmp_predict[:, :2], tmp_predict[:, 2])
                    COn = COn + score['Onset_F-measure']
                    COnP = COnP + score['F-measure_no_offset']
                    COnPOff = COnPOff + score['F-measure']
                    weighted_f1 = COn * 0.2 + COnP * 0.6 + COnPOff * 0.2

                
            # if batch_idx % 50 == 0:
            #     print ("epoch %d, sample %d, loss %.6f" %(epoch, batch_idx, total_loss))

        print('epoch %d, total loss: %.6f, training time: %.3f sec' %(epoch, train_loss/ total_length, time.time()-start_time))
        
        if epoch >= 10:
            print ("epoch %d, COn %.6f, COnP %.6f, COnPOff %.6f, Training weighted_f1 %.6f" %(epoch, COn/ count, 
                COnP/ count, COnPOff/ count, weighted_f1/ count))

        
        # evaluate
        if epoch >= 10:
            val_total_loss = 0
            COn = 0
            COnP = 0
            COnPOff = 0
            weighted_f1 = 0
            count = 0
            on_net.eval()
            off_net.eval()
            for idx, sample in enumerate(val_loader):
                data = torch.Tensor(sample['data'])
                target = torch.Tensor(sample['label'])
                data_lens = sample['data_lens']
                vocal_pitch = sample['vocal_pitch']
                _range = sample['range']
                
                data_length= list(data.shape)[0]

                data = data.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)

                on_output = on_net(data)
                off_output = off_net(data)

                on_loss = criterion_set(on_output, torch.narrow(target, dim= 2, start= 0, length= 1))
                off_loss = criterion_set(off_output, torch.narrow(target, dim= 2, start= 1, length= 1))
                set_loss = on_loss + off_loss

                val_total_loss += set_loss.item()

                output = torch.cat([on_output, off_output], dim=2)
                predict = post_processing(output, vocal_pitch, _range)

                gt = np.array(sample['groundtruth'])
                #predict = np.array(predict)          

                for i in range(len(predict)):
                    count += 1
                    tmp_predict = np.where(np.array(predict[i]) > 0, np.array(predict[i]), 0.0001)
                    if tmp_predict.shape[0] == 0:
                        continue
                    score = mir_eval.transcription.evaluate(gt[i][:, :2], gt[i][:, 2], tmp_predict[:, :2], tmp_predict[:, 2])
                    COn = COn + score['Onset_F-measure']
                    COnP = COnP + score['F-measure_no_offset']
                    COnPOff = COnPOff + score['F-measure']
                    weighted_f1 = COn * 0.2 + COnP * 0.6 + COnPOff * 0.2 

            scheduler.step(weighted_f1)

            if weighted_f1 > best_f1:
                best_f1 = weighted_f1
                #model_path= f'ST_{epoch}.pt'
                torch.save(on_net.state_dict(), './onset_model.pkl')
                torch.save(off_net.state_dict(), './offset_model.pkl')
            
            if epoch >= 10:
                print('epoch %d, total val loss: %.6f ' %(epoch, val_total_loss/ total_length))

                print ("epoch %d, COn %.6f, COnP %.6f, COnPOff %.6f, Validation weighted_f1 %.6f" %(epoch, COn/ count,
                 COnP/ count, COnPOff/ count, weighted_f1/ count))

    return net
