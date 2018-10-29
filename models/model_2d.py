# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:48:09 2018

basic convolutional model for hand track classification

@author: Γιώργος
"""

import torch.nn as nn

def xavier(net):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 and hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight.data, gain=1.)
            print(classname, "initialized with xavier.")
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.fill_(1.0)
            print(classname, "filled with ones.")
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight.data, gain=1.)
            print(classname, "initialized with xavier.")
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname in ['Sequential', 'AvgPool3d', 'MaxPool3d', \
                           'Dropout', 'ReLU', 'Softmax', 'BnActConv3d'] \
             or 'Block' in classname:
            print(classname, "uninitialized on purpose.")
        else:
            if classname != classname.upper():
                print("Initializer:: '{}' is uninitialized.".format(classname))
    net.apply(weights_init)


class Conv2dBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(Conv2dBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv(x)
        h = self.relu(self.bn(h))
        return h

class Basic2DConvNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(Basic2DConvNet, self).__init__()

        self.conv1 = Conv2dBNRelu(3, 16, kernel=(7,7), stride=(2,2), padding=(3,3))
        self.pool1 = nn.MaxPool2d((3,3), stride=(2,2))

        self.conv2 = Conv2dBNRelu(16, 96, kernel=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = Conv2dBNRelu(96, 192, kernel=(3,3), stride=(2,2), padding=(1,1))
        self.conv4 = Conv2dBNRelu(192, 192, kernel=(1,1), stride=(1,1), padding=0)
        self.conv5 = Conv2dBNRelu(192, 384, kernel=(3,3), stride=(2,2), padding=(1,1))
        self.conv6 = Conv2dBNRelu(384, 768, kernel=(3,3), stride=(2,2), padding=(1,1))

        self.pool2 = nn.AvgPool2d((14, 8))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(768, num_classes)

        xavier(net=self)

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.pool2(h)
        h = self.dropout(h)

        h = h.view(h.shape[0], -1)
        h = self.linear(h)

        return h

if __name__ == "__main__":
    import torch
    # ---------
    net = Basic2DConvNet(num_classes=100, dropout=0.5)
    data = torch.tensor(torch.randn(1,3,456,256))
    output = net(data)
#    torch.save({'state_dict': net.state_dict()}, './tmp.pth')
    print (output.shape)
