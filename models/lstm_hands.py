# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:17:49 2018

define lstm model

@author: Γιώργος
"""

import torch
import torch.nn as nn

class LSTM_Hands(nn.Module):
    # source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_Hands, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            bias=True, batch_first=False, dropout=0, bidirectional=False)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.num_h_features),
                torch.zeros(1, 1, self.num_h_features))
        
    def forward(self, seq_batch_coords, seq_lengths):
        # seq_batch_coords is sorted descending in sequence size so we can pad
#        lstm_input = nn.utils.rnn.nn
        batch_size = seq_batch_coords.size(1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        
        seq_batch_coords = nn.utils.rnn.pack_padded_sequence(seq_batch_coords, seq_lengths)
        
        out, _ = self.lstm(seq_batch_coords, (h0, c0))
        
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        
        # get the state of the hidden before the padded inputs start
        out = out[seq_lengths-1, list(range(batch_size)), :]
        out = self.fc(out)
        
        return out
    