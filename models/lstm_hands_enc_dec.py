# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:20:51 2018

lstm encoder decoder for hands

@author: Γιώργος
"""

import torch
import torch.nn as nn

from utils.file_utils import print_and_save

class LSTM_Hands_encdec(nn.Module):
    # source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
    def __init__(self, input_size, row_hidden, time_hidden, num_layers, num_classes, dropout, log_file=None):
        super(LSTM_Hands_encdec, self).__init__()
        self.row_hidden = row_hidden
        self.time_hidden = time_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.log_file=log_file
        
        self.row_lstm = nn.LSTM(input_size, row_hidden, 1, 
                               bias=True, batch_first=False, dropout=dropout, bidirectional=False)
        
        self.time_lstm = nn.LSTM(row_hidden, time_hidden, num_layers,
                                 bias=True, batch_first=False, dropout=dropout, bidirectional=False)

        self.fc = nn.Linear(time_hidden, num_classes)
        
        
    def forward(self, seq_height_width, seq_lengths): 
        # seq_batch_coords 256, x, 456

        h0_row = torch.zeros(1, 1, self.row_hidden).cuda()
        c0_row = torch.zeros(1, 1, self.row_hidden).cuda()
        
        h0_time = torch.zeros(self.num_layers, 1, self.time_hidden).cuda()
        c0_time = torch.zeros(self.num_layers, 1, self.time_hidden).cuda()
        
        im_hiddens = []        
        for i in range(seq_height_width.size(0)):
            row_out, _ = self.row_lstm(seq_height_width[i].unsqueeze(1), (h0_row, c0_row))
            im_hiddens.append(row_out[-1]) # can also concatenate the hiddens for an image
        
        time_input = torch.stack(im_hiddens)#.unsqueeze(1)
        time_out, _ = self.time_lstm(time_input, (h0_time, c0_time))
        
        out = self.fc(time_out[-1])
        
        return out