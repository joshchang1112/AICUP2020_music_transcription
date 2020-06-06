import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import mir_eval
import time
from utils import post_processing

def do_training(net, train_loader, val_loader, device):

    num_epoch = 50
    criterion_onset = nn.BCELoss()
    criterion_pitch = nn.SmoothL1Loss()
    #criterion_pitch = nn.CrossEntropyLoss()
    train_loss= 0.0
    total_length= 0
    best_f1 = 0

    net_1 = net[0]
    optimizer_1 = optim.Adam(net_1.parameters(), lr= 4e-3)
    scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, 50 * 5, 0.9)

    net_2 = net[1]
    optimizer_2 = optim.Adam(net_2.parameters(), lr= 8e-3)
    scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, 50 * 10, 0.5)

    for epoch in range(num_epoch):
        total_length= 0.0
        start_time = time.time()
        train_loss= 0.0
        total_set_loss = 0.0
        total_pitch_loss=0.0

        COn = 0
        COnP = 0
        COnPOff = 0
        weighted_f1 = 0
        count = 0 
        # for param_group in optimizer.param_groups:
        #     print("lr: {}".format(param_group['lr']))


        net_1.train()
        net_2.train()
        for batch_idx, sample in enumerate(train_loader):

            # print(len(sample['label']))
            # print(sample['label'][0].shape)
            # print(sample['label'][1].shape)
            data = sample['data']
            target = sample['label']
            data_lens = sample['data_lens']

            data_length= list(data.shape)[0]

            data = data.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            optimizer_1.zero_grad()
            output1 = net_1(data)
            set_loss = criterion_onset(output1, torch.narrow(target, dim= 2, start= 0, length= 2))
            set_loss.backward()
            optimizer_1.step()
            scheduler_1.step()

            optimizer_2.zero_grad()
            output2 = net_2(data)
            pitch_loss = criterion_pitch(output2, torch.narrow(target, dim= 2, start= 2, length= 1))
            pitch_loss.backward()
            optimizer_2.step()
            scheduler_2.step()
            
            total_set_loss += set_loss.item()
            total_pitch_loss += pitch_loss.item()
            train_loss += (set_loss.item() + pitch_loss.item())
            total_length = total_length + 1
            
            if epoch >= 10:
                gt = np.array(sample['groundtruth'])
                predict = post_processing(output1, output2)
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

        print('epoch %d, avg loss: %.6f, set loss: %.6f, pitch loss: %.6f, training time: %.3f sec' %(epoch, train_loss/ total_length, \
            total_set_loss/ total_length, total_pitch_loss/ total_length, time.time()-start_time))
        
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
            net_1.eval()
            net_2.eval()
            for idx, sample in enumerate(val_loader):
                val_loss= 0.0
                data = torch.Tensor(sample['data'])
                target = torch.Tensor(sample['label'])
                data_lens = sample['data_lens']

                data_length= list(data.shape)[0]

                data = data.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)

                output1 = net_1(data)
                output2 = net_2(data)

                # total_loss = criterion_onset(output1, torch.narrow(target, dim= 2, start= 0, length= 2))
                # total_loss = total_loss + criterion_pitch(output2, torch.narrow(target, dim= 2, start= 2, length= 1))
                # val_loss += total_loss.item()
                predict = post_processing(output1, output2)

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

        
            if weighted_f1 / len(val_loader) > best_f1:
                best_f1 = weighted_f1 / len(val_loader)
                model_path= f'ST_{epoch}.pt'
                torch.save(net_1.state_dict(), './model_1' + model_path)
                torch.save(net_2.state_dict(), './model_2' + model_path)
            if epoch >= 10:
                print ("epoch %d, COn %.6f, COnP %.6f, COnPOff %.6f, Validation weighted_f1 %.6f" %(epoch, COn/ count,
                 COnP/ count, COnPOff/ count, weighted_f1/ count))   

    return net_1, net_2