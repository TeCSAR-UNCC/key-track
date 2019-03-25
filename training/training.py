from __future__ import print_function
import numpy as np
import random


import os.path
import datetime
import utils 

import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
'''     Device configuration      '''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

''' HYPERPARAMETERS'''
#num_steps = 6
input_size = 54
hidden_size = 64
num_layers=1
num_classes = 4
batch_size = 1
epoches =  75
learning_rate = 0.000001
num_videos = 64  
path_to_data = sorted(os.listdir('../duke_dataset/training/'))



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self,input_x):

        #Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size).to(device)

        #Forward propagate LSTM
        #out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(input_x,(h0,c0))

        out = self.fc(out[:,-1,:])
        return out

#vary time step

tstep=[3]

for num_steps in tstep:
    print('TRAINING for step '+str(num_steps))
    model = RNN(input_size,hidden_size,num_layers,num_classes).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    iters = num_videos * epoches


    vid_list=random.sample(range(0,len(path_to_data),2),num_videos)

    per_epoch_loss=[]
    start_time = time.time()
    for epoch in range(epoches):
        cnt=0
        total_loss=0
        for i in (vid_list):
            #i = i % num_videos
            #in ROLO utils file, the sequence for MOT16 starts with 30 Modification 3

            [w_img, h_img, sequence_name ]= 1920.0,1080.0,path_to_data[i]

            x_path = os.path.join('../duke_dataset/training/', sequence_name) #Modification 4
            #x_bbox_path = os.path.join('own_images/training/', sequence_name, 'bbox_video.npy')

            y_path = '../duke_dataset/training/'+ sequence_name[:-4]+ '_gt.npy' ##Modification 5
            #print y_path

            filegt = np.load(y_path)
            filefeatures = np.load(x_path)

            training_iters = filefeatures.shape[0]                
            #filebboxes = np.load(x_bbox_path)

            print('Sequence '+ sequence_name+' chosen')
            id =0
            video_loss=0
            while id  < training_iters - num_steps*batch_size:

                # Load training data & ground truth
                batch_input = utils.load_openpose_features_train(w_img,h_img,filefeatures , batch_size, num_steps, id) # [num_of_examples, input_size] (depth == 1)

                batch_groundtruth = utils.load_openpose_gt_train(w_img,h_img,filegt, batch_size, num_steps, id)


                batch_input = np.reshape(batch_input, [batch_size, num_steps, input_size])

                batch_input = (torch.from_numpy(batch_input)).to(device)
                #print(batch_input, batch_input.shape)
                batch_groundtruth = np.reshape(batch_groundtruth, [batch_size, num_classes]) #2*4

                batch_groundtruth = (torch.from_numpy(batch_groundtruth)).to(device)
                #print(batch_groundtruth , batch_groundtruth.shape)        
                outputs = model(batch_input)
                loss = criterion(outputs,batch_groundtruth)*100

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print(id)
                video_loss += loss.item()
                id = id +1
            
            print ('Epoch [{}/{}], Video [{}/{}], Loss: {:.6f}'.format(epoch+1, epoches, cnt+1, num_videos, video_loss/id))
            print ('\n')
            cnt+=1
            total_loss+= (video_loss/id)
            #print('Sequence '+sequence_name+' done')
        per_epoch_loss.append(total_loss/cnt)
    end_time = time.time()

    print('Time for '+str(num_steps)+ ' is: '+str(((end_time-start_time)/60.0)/60.0) + ' hours')
    
    #np.save('model3_epoch75/step'+str(num_steps)+'loss',np.asarray(per_epoch_loss))
    np.savetxt('model3_epoch75/step'+str(num_steps)+'loss.csv',np.asarray(per_epoch_loss))
    torch.save(model.state_dict(), 'model3_epoch75/model_step'+str(num_steps)+'_.ckpt')
    print('Model for step '+str(num_steps)+' saved')

