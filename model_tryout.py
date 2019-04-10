# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 10:51:58 2019

model tryout with dummy data to see how it works

@author: Γιώργος
"""

import os
import torch
import torch.nn as nn
import numpy as np
from utils.file_utils import print_and_save
from utils.dataset_loader import calc_angles, calc_polar_distance_from_prev, load_pickle
import cv2

fr_width = 456
fr_height = 256
split_color = (1.,1.,1.)

def visualize_points(points):
    left_track = (points[:,:2] * [456,256]).astype(np.int)
    right_track = (points[:,4:6] * [456,256]).astype(np.int)
    
    dir_base= r"D:\imgs"
    os.mkdir(dir_base)
    
    num_points = len(left_track)
    image = np.ones([fr_height,fr_width,3])
    for i, (left, right) in enumerate(zip(left_track, right_track)):
        left_color = (1., i/num_points, 0.)
        right_color = (1., 0., i/num_points)
        if left[0] < fr_width and left[1] < fr_height:                
            image[int(left[1]), int(left[0])] = left_color
        if right[0] < fr_width and right[1] < fr_height:
            image[int(right[1]), int(right[0])] = right_color
        
#        cv2.imshow("tracks full", cv2.resize(image, (456*2, 256*2)))
#        cv2.waitKey(0)
        
        cv2.imwrite(os.path.join(dir_base, "frame_{:010d}.jpg".format(i)), np.array(image[100:,100:]*255, dtype=np.uint8))
    
def visualize_prediction(points, outputs):
    left_track = (points[:,:2] * [456,256]).astype(np.int)
    left_dist = points[:,2]
    left_angles = points[:,3]
    right_track = (points[:,4:6] * [456,256]).astype(np.int)
    right_dist = points[:,6]
    right_angle = points[:,7]
    
    num_points = len(left_track)
    image = np.zeros([fr_height+10,fr_width,3])
    for i, (left, right, output) in enumerate(zip(left_track, right_track, outputs)):
        image[fr_height:fr_height+10,:,:] = np.zeros((10,fr_width,3))
        left_color = (1., i/num_points, 0.)
        right_color = (1., 0., i/num_points)
        if left[0] < fr_width and left[1] < fr_height:                
            image[int(left[1]), int(left[0])] = left_color
        if right[0] < fr_width and right[1] < fr_height:
            image[int(right[1]), int(right[0])] = right_color

        top5 = output.topk(5)[1].detach().cpu().numpy()[0]
        top5_txt = "{} {} {} {} {}".format(top5[0], top5[1], top5[2],top5[3],top5[4])
        cv2.putText(image, top5_txt, (10, fr_height+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, split_color,1)
        cv2.imshow("tracks full", cv2.resize(image, (456*2, 256*2)))
        cv2.waitKey(1)
#        cv2.imwrite(os.path.join("", "frame_{:010d}.jpg".format(i)), np.array(image*255, dtype=np.uint8))
#%%
class LSTM_Hands_Polar(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, **kwargs):
        super(LSTM_Hands_Polar, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = kwargs.get('dropout')
        self.bidir = kwargs.get('bidir')
        self.log_file= kwargs.get('log_file')
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            bias=True, batch_first=False, dropout=0.0, bidirectional=self.bidir)
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, seq_batch_coords, seq_lengths):
        batch_size = seq_batch_coords.size(1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        
#        packed_inputs = nn.utils.rnn.pack_padded_sequence(seq_batch_coords, seq_lengths)
#        lstm_out, hidden = self.lstm(packed_inputs, (h0, c0))
#        unpacked_out, dunno = nn.utils.rnn.pad_packed_sequence(lstm_out)
##       get the state of the hidden before the padded inputs start
#        out = unpacked_out[seq_lengths-1, list(range(batch_size)), :]
        
        lstm_out, hidden = self.lstm(seq_batch_coords, (h0, c0))
        outs = []
        for lstm_seq_pred in lstm_out:
            out = self.fc(lstm_seq_pred)
            outs.append(out)
        return outs
    
    def forward_bidir(self, seq_batch_coords, seq_lengths):
        batch_size = seq_batch_coords.size(1)
        h0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).cuda()
        
        lstm_out, hidden = self.lstm(seq_batch_coords, (h0, c0))
        concats = []
        for i in range(len(lstm_out)):
            lstm_out_for = lstm_out[i,:,:self.hidden_size//2]
            lstm_out_back = lstm_out[len(lstm_out)-1-i, :, self.hidden_size//2:]
            concats.append(torch.cat(lstm_out_for, lstm_out_back),dim=-1)
        outs = []
        for conc in concats:
            out = self.fc(conc)
            outs.append(out)
        return outs

def load_multiple_tracks(paths):
    lefts = []
    rights = []
    for p in paths:
        hnd_trc = load_pickle(p)
        lefts += hnd_trc['left']
        rights += hnd_trc['right']
    left_track = np.array(lefts, dtype=np.float32)
    right_track = np.array(rights, dtype=np.float32)
    return left_track, right_track
    
log_file=None
no_norm_input = False
ckpt_path = r"outputs\lstm_polar_128_0.0_1000_8_32_2_seq32_coords_polar_clr_tri_vsel125\lstm_polar_128_0.0_1000_8_32_2_seq32_coords_polar_clr_tri_vsel125_best.pth"
val_list = r"splits\hand_tracks\hand_locs_val_1.txt"
lstm_input, lstm_hidden, lstm_layers, verb_classes, lstm_seq_size = 8, 32, 2, 125, 32

lstm_model = LSTM_Hands_Polar
kwargs = {'dropout': 0, 'bidir':True}    
model_ft = lstm_model(lstm_input, lstm_hidden, lstm_layers, verb_classes, **kwargs)
model_ft = torch.nn.DataParallel(model_ft).cuda()
checkpoint = torch.load(ckpt_path)    
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft.eval()
print_and_save("Model loaded to gpu", log_file)
criterion=torch.nn.CrossEntropyLoss().cuda()

#%%
norm_val = [1., 1., 1., 1.] if no_norm_input else [456., 256., 456., 256.]
norm_val = np.array(norm_val)
#dataset_loader = PointPolarDatasetLoader(val_list, max_seq_length=lstm_seq_size,
#                                         norm_val=norm_val, validation=True)

#track_path = r"D:\Code\epic-kitchens-processing\output\yolo_allhands_tracked_videos\clean\P01\P01_04.pkl"
#track_path = r"D:\Datasets\egocentric\EPIC_KITCHENS\clean_hand_detection_tracks\P30\P30_05\179200_5_140.pkl"
track_path = r"D:\Datasets\egocentric\EPIC_KITCHENS\clean_hand_detection_tracks\P30\P30_05\81347_4_11.pkl"
hand_tracks = load_pickle(track_path)
left_track = np.array(hand_tracks['left'], dtype=np.float32)
right_track = np.array(hand_tracks['right'], dtype=np.float32)

#trc_base = r"D:\Datasets\egocentric\EPIC_KITCHENS\clean_hand_detection_tracks\P28\P28_09"
#path_names = ["14932_5_30.pkl","15207_5_30.pkl","15493_5_30.pkl","15493_5_30.pkl",
#              "15719_5_30.pkl","15734_5_30.pkl","15961_5_30.pkl","15988_5_30.pkl",
#              "16232_5_30.pkl","16464_5_30.pkl","17158_5_30.pkl","17426_5_30.pkl"]
#paths = [os.path.join(trc_base, x) for x in path_names]
#left_track, right_track = load_multiple_tracks(paths)

left_track = left_track[np.linspace(0, len(left_track), 192, endpoint=False, dtype=int)]
right_track = right_track[np.linspace(0, len(right_track), 192, endpoint=False, dtype=int)]
left_angles = calc_angles(left_track)
right_angles = calc_angles(right_track)
left_track /= norm_val[:2]
right_track /= norm_val[2:]
left_dist = calc_polar_distance_from_prev(left_track)
right_dist = calc_polar_distance_from_prev(right_track)
points = np.concatenate((left_track,
                         left_dist[:, np.newaxis],
                         left_angles[:, np.newaxis],
                         right_track,
                         right_dist[:, np.newaxis],
                         right_angles[:, np.newaxis]), -1).astype(np.float32)

#%%
with torch.no_grad():
    inputs = torch.tensor(points).cuda().unsqueeze_(0)
    targets = torch.tensor(np.array([9], dtype=np.int64)).cuda()
    
    inputs = inputs.transpose(1,0)
    output = model_ft(inputs, torch.tensor([points.shape[0]]))
    for out in output:
        print(out.topk(5)[0].cpu().detach().numpy(), out.topk(5)[1].cpu().detach().numpy())
#        print(out.topk(5)[1].cpu().detach().numpy())
#    loss = criterion(output, targets)
visualize_prediction(points, output)

#%%
visualize_points(points)