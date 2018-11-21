# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:55:53 2018

Image dataset loader for a .txt file with a sample per line in the format
'path of image start_frame verb_id noun_id'

@author: Γιώργος
"""

import pickle
import cv2
import numpy as np
import torch.utils.data


def parse_samples_list(list_file):
    return [ImageData(x.strip().split(' ')) for x in open(list_file)]

def load_pickle(tracks_path):
    with open(tracks_path,'rb') as f:
        tracks = pickle.load(f)
    return tracks

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
                 batch_transform=None, channels='RGB', validation=False ):
        self.samples_list = parse_samples_list(list_file)
        self.transform = batch_transform
        self.channels = channels
        self.validation = validation
        self.image_read_type = cv2.IMREAD_COLOR if channels=='RGB' else cv2.IMREAD_GRAYSCALE

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        img = cv2.imread(self.samples_list[index].image_path, self.image_read_type).astype(np.float32)
        if self.channels=='RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if not self.validation:
            return img, self.samples_list[index].label_verb
        else:
            name_parts = self.samples_list[index].image_path.split("\\")
            return img, self.samples_list[index].label_verb, name_parts[-2] + "\\" + name_parts[-1]

class PointDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, list_file, batch_transform=None, norm_val=[1.,1.,1.,1.], 
                 validation=False):
        self.samples_list = parse_samples_list(list_file)
        self.transform = batch_transform
        self.norm_val = np.array(norm_val)
        self.validation = validation
    
    def __len__(self):
        return len(self.samples_list)
    
    def __getitem__(self, index):
        hand_tracks = load_pickle(self.samples_list[index].image_path)
        
        left_track = np.array(hand_tracks['left'], dtype=np.float32)
        right_track = np.array(hand_tracks['right'], dtype=np.float32)
        points = np.concatenate((left_track, right_track), -1)
        
        points /= self.norm_val
        if not self.validation:
            return points, len(points), self.samples_list[index].label_verb
        else:
            name_parts = self.samples_list[index].image_path.split("\\")
            return points, len(points), self.samples_list[index].label_verb, name_parts[-2] + "\\" + name_parts[-1]

class PointVectorSummedDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, list_file, validation=False):
        self.samples_list = parse_samples_list(list_file)
        self.validation = validation
        
    def __len__(self):
        return len(self.samples_list)
    
    def __getitem__(self, index):
        hand_tracks = load_pickle(self.samples_list[index].image_path)
        left_track = np.array(hand_tracks['left'], dtype=np.int)
        right_track = np.array(hand_tracks['right'], dtype=np.int)   
        seq_size = len(left_track)
        
        vec = np.zeros((seq_size, 456+256), dtype=np.float32)
        for i, (x, y) in enumerate(left_track):
            if y < 256:
                vec[:i, x] += 1
                vec[:i, 456+y] += 1
        for i, (x, y) in enumerate(right_track):
            if y < 256:
                vec[:i, x] += 1
                vec[:i, 456+y] += 1
                
        if not self.validation:
            return vec, seq_size, self.samples_list[index].label_verb
        else:
            name_parts = self.samples_list[index].image_path.split("\\")
            return vec, seq_size, self.samples_list[index].label_verb, name_parts[-2] + "\\" + name_parts[-1]
                

class PointImageDatasetLoader(torch.utils.data.Dataset):
    sum_seq_size = 0
    def __init__(self, list_file, batch_transform=None, norm_val=[1.,1.,1.,1.],
                 validation=False):
        self.samples_list = parse_samples_list(list_file)
        self.transform = batch_transform
        self.norm_val = np.array(norm_val)
        self.validation = validation
    
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        hand_tracks = load_pickle(self.samples_list[index].image_path)
        left_track = np.array(hand_tracks['left'], dtype=np.int)
        right_track = np.array(hand_tracks['right'], dtype=np.int)
    
        seq_size = len(left_track)
#        print(seq_size)
        self.sum_seq_size += seq_size
        print(self.sum_seq_size)
        point_imgs = np.zeros([385, 456, seq_size], dtype=np.float32)
        
        
        for i in range(seq_size):
            intensities = np.linspace(1., 0., i+1)[::-1]
            point_imgs[left_track[:i+1,1], left_track[:i+1,0], i] = intensities
            point_imgs[right_track[:i+1,1], right_track[:i+1,0], i] = intensities
        
#        for i in range(seq_size):
#            for j in range(i+1):
#                xl, yl = left_track[j]
#                xr, yr = right_track[j]
#                if xl < 456 and yl < 256:
#                    point_imgs[int(yl), int(xl), i] = (j+1)/(i+1)    
#                if xr < 456 and yr < 256:
#                    point_imgs[int(yr), int(xr), i] = (j+1)/(i+1)
#                cv2.imshow('1', point_imgs[:,:,i])
#                cv2.waitKey(5)
#        cv2.waitKey(0)
        
        return point_imgs[:256, :, :], seq_size, self.samples_list[index].label_verb
        
if __name__=='__main__':
    
    from dataset_loader_utils import Resize, ResizePadFirst
    
    image = cv2.imread(r"..\hand_detection_track_images\P24\P24_08\90442_0_35.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)

    resize_only = Resize((224,224), False, cv2.INTER_CUBIC)
#    resize_pad = ResizeZeroPad(224, True, cv2.INTER_NEAREST)
    cubic_pf_fun = ResizePadFirst(224, True, cv2.INTER_CUBIC)
    linear_pf_fun = ResizePadFirst(224, True, cv2.INTER_LINEAR)
    nearest_pf_fun = ResizePadFirst(224, True, cv2.INTER_NEAREST)
    area_pf_fun = ResizePadFirst(224, True, cv2.INTER_AREA)
    lanc_pf_fun = ResizePadFirst(224, True, cv2.INTER_LANCZOS4)
    linext_pf_fun = ResizePadFirst(224, True, cv2.INTER_LINEAR_EXACT)

    resize_nopad = resize_only(image)
#    resize_pad_first = resize_pad(image)
    
    cubic_pf = cubic_pf_fun(image)
#    cubic_pf = np.where(cubic_pf_fun(image) > 1, 255, 0).astype(np.float32)
#    nearest_pf = np.where(nearest_pf_fun(image) > 1, 255, 0).astype(np.uint8)
#    linear_pf = np.where(linear_pf_fun(image) > 1, 255 ,0).astype('uint8')
#    area_pf = np.where(area_pf_fun(image) > 1, 255 ,0).astype('uint8')
#    lanc_pf = np.where(lanc_pf_fun(image) > 1, 255, 0).astype('uint8')
    linext_pf = linext_pf_fun(image)
    
    cv2.imshow('original', image)
    cv2.imshow('original resize', resize_nopad)
#    cv2.imshow('padded resize', resize_pad_first)
    cv2.imshow('cubic', cubic_pf)
#    cv2.imshow('nearest', nearest_pf)
#    cv2.imshow('area', area_pf)
#    cv2.imshow('lanc', lanc_pf)
    cv2.imshow('linext', linext_pf)    
    
    cv2.waitKey(0)
    
    