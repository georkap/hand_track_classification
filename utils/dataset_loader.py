# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:55:53 2018

Image dataset loader for a .txt file with a sample per line in the format
'path of image start_frame verb_id noun_id'

@author: Γιώργος
"""

import cv2
import numpy as np
import torch.utils.data
import torchvision.transforms

def parse_samples_list(list_file):
    return [ImageData(x.strip().split(' ')) for x in open(list_file)]

class ImageData(object):

    def __init__(self, row):
        self.data = row

    @property
    def image_path(self):
        return self.data[0]

    @property
    def num_frames(self):
        return int(self.data[1])

    @property
    def label_verb(self):
        return int(self.data[2])

    @property
    def label_noun(self):
        return int(self.data[3])

class DatasetLoader(torch.utils.data.Dataset):

    def __init__(self, list_file,
                 batch_transform=None, channels='RGB' ):
        self.samples_list = parse_samples_list(list_file)
        self.transform = batch_transform
        self.channels = channels
        self.image_read_type = cv2.IMREAD_COLOR if channels=='RGB' else cv2.IMREAD_GRAYSCALE

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        img = cv2.imread(self.samples_list[index].image_path, self.image_read_type).astype(np.float32)
        if self.channels=='RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img /= 255.

        if self.transform is not None:
            img = self.transform(img)

        return img, self.samples_list[index].label_verb

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    """
    def __init__(self):
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        if self.rng.rand() < 0.5:
            data = np.fliplr(data)
            data = np.ascontiguousarray(data)
        return data

class ResizePadFirst(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size # int
        self.interpolation = interpolation    

    def __call__(self, data):
        h, w, c = data.shape if len(data.shape) == 3 else (data.shape[0], data.shape[1],1)
        
        assert isinstance(self.size, int)
        
        largest = w if w > h else h
        delta_w = largest - w
        delta_h = largest - h
        
        top, bottom = delta_h//2, delta_h - delta_h//2
        left, right = delta_w//2, delta_w - delta_w//2
        color = [0] * c
        
        padded_data = cv2.copyMakeBorder(data, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
              
        scaled_data = cv2.resize(padded_data, (self.size, self.size), interpolation=self.interpolation)
        
        scaled_data = np.where(scaled_data > 1, 255, 0).astype(np.float32)
        return scaled_data


class ResizeZeroPad(object):
    """ Rescales the input numpy array to the given 'size'
    and zero pads to fill the greater dimension. 
    For example, if height > width, the image will be 
    rescaled to height == size and the remaining values of
    width will be zeros. 
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size # int
        self.interpolation = interpolation
    
    def __call__(self, data):
        h, w, c = data.shape if len(data.shape) == 3 else (data.shape[0], data.shape[1],1)
        
        assert isinstance(self.size, int)
        
        if w < h:
            new_w = int(self.size * w / h)
            new_h = self.size
        else:
            new_w = self.size
            new_h = int(self.size * h / w)
            
        if (h != new_h) or (w != new_w):
            scaled_data = cv2.resize(data, (new_w, new_h), interpolation=self.interpolation)
        else:
            scaled_data = data
            
        delta_w = self.size - new_w
        delta_h = self.size - new_h
        top, bottom = delta_h//2, delta_h - delta_h//2
        left, right = delta_w//2, delta_w - delta_w//2
        
        color = [0] * c
        
        padded_data = cv2.copyMakeBorder(scaled_data, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return padded_data

class Resize(object):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size # [w, h]
        self.interpolation = interpolation

    def __call__(self, data):
        h, w, c = data.shape if len(data.shape) == 3 else (data.shape[0], data.shape[1],1)

        if isinstance(self.size, int):
            slen = self.size
            if min(w, h) == slen:
                return data
            if w < h:
                new_w = self.size
                new_h = int(self.size * h / w)
            else:
                new_w = int(self.size * w / h)
                new_h = self.size
        else:
            new_w = self.size[0]
            new_h = self.size[1]

        if (h != new_h) or (w != new_w):
            scaled_data = cv2.resize(data, (new_w, new_h), interpolation=self.interpolation)
        else:
            scaled_data = data

        return scaled_data

class RandomCrop(object):
    """Crops the given numpy array at the random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape
        th, tw = self.size
        x1 = self.rng.choice(range(w - tw))
        y1 = self.rng.choice(range(h - th))
        cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
        return cropped_data
    
    
if __name__=='__main__':
    image = cv2.imread(r"..\hand_detection_track_images\P24\P24_08\90442_0_35.png", cv2.IMREAD_GRAYSCALE)
    resize_only = Resize((224,224), cv2.INTER_NEAREST)
    resize_pad = ResizeZeroPad(224, cv2.INTER_NEAREST)
    cubic_pf_fun = ResizePadFirst(224, cv2.INTER_CUBIC)
    linear_pf_fun = ResizePadFirst(224, cv2.INTER_LINEAR)
    nearest_pf_fun = ResizePadFirst(224, cv2.INTER_NEAREST)
    area_pf_fun = ResizePadFirst(224, cv2.INTER_AREA)
    lanc_pf_fun = ResizePadFirst(224, cv2.INTER_LANCZOS4)
    linext_pf_fun = ResizePadFirst(224, cv2.INTER_LINEAR_EXACT)

    resize1 = resize_only(image)
    resize2 = resize_pad(image)
    
    cubic_pf = cubic_pf_fun(image)
#    cubic_pf = np.where(cubic_pf_fun(image) > 1, 255, 0).astype(np.float32)
#    nearest_pf = np.where(nearest_pf_fun(image) > 1, 255, 0).astype(np.uint8)
#    linear_pf = np.where(linear_pf_fun(image) > 1, 255 ,0).astype('uint8')
#    area_pf = np.where(area_pf_fun(image) > 1, 255 ,0).astype('uint8')
#    lanc_pf = np.where(lanc_pf_fun(image) > 1, 255, 0).astype('uint8')
    linext_pf = linext_pf_fun(image)
    
    cv2.imshow('original', image)
    cv2.imshow('original resize', resize1)
    cv2.imshow('padded resize', resize2)
    cv2.imshow('cubic', cubic_pf)
#    cv2.imshow('nearest', nearest_pf)
#    cv2.imshow('area', area_pf)
#    cv2.imshow('lanc', lanc_pf)
    cv2.imshow('linext', linext_pf)    
    
    cv2.waitKey(0)
    
    