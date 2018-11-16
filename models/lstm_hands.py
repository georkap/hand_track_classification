# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:17:49 2018

define lstm model

@author: Γιώργος
"""

import torch
import torch.nn as nn

from utils.file_utils import print_and_save

class LSTM_Hands(nn.Module):
    # source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, log_file=None):
        super(LSTM_Hands, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.log_file=log_file
        
#        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
#                            bias=True, batch_first=False, dropout=dropout, bidirectional=False)
#        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bias=True,
#                          batch_first=False, dropout=dropout, bidirectional=False)
        self.left_lstm = nn.LSTM(int(input_size/2), int(hidden_size/2), num_layers,
                                 bias=True, batch_first=False, dropout=dropout,
                                 bidirectional=False)
        self.right_lstm = nn.LSTM(int(input_size/2), int(hidden_size/2), num_layers,
                                 bias=True, batch_first=False, dropout=dropout,
                                 bidirectional=False)        
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
        
    def forward(self, seq_batch_coords, seq_lengths):
        # seq_batch_coords is sorted descending in sequence size so we can pad
#        lstm_input = nn.utils.rnn.nn
        batch_size = seq_batch_coords.size(1)
#        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
#        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        
#        packed_inputs = nn.utils.rnn.pack_padded_sequence(seq_batch_coords, seq_lengths)
#        lstm_out, _ = self.lstm(packed_inputs, (h0, c0))
#        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out)
#        rnn_out, _ = self.rnn(packed_inputs, h0)
#        unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out)

        # get the state of the hidden before the padded inputs start
#        out = unpacked_out[seq_lengths-1, list(range(batch_size)), :]
        
        # for dual lstm
        h0 = torch.zeros(self.num_layers, batch_size, int(self.hidden_size/2)).cuda()
        c0 = torch.zeros(self.num_layers, batch_size, int(self.hidden_size/2)).cuda()
        
        left_packed = nn.utils.rnn.pack_padded_sequence(seq_batch_coords[:,:,:2], seq_lengths)
        right_packed = nn.utils.rnn.pack_padded_sequence(seq_batch_coords[:,:,2:], seq_lengths)
        left_lstm_out, _ = self.left_lstm(left_packed, (h0, c0))
        right_lstm_out, _ = self.right_lstm(right_packed, (h0, c0))
        left_unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(left_lstm_out)
        right_unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(right_lstm_out)
        
        left_out = left_unpacked_out[seq_lengths-1, list(range(batch_size)), :]
        right_out = right_unpacked_out[seq_lengths-1, list(range(batch_size)), :] 
        
        out = torch.cat((left_out, right_out), dim=-1)
        if self.log_file:
            print_and_save(out, self.log_file)
        out = self.fc(out)
        
        return out
    