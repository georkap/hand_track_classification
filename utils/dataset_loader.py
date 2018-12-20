# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:55:53 2018

Image dataset loader for a .txt file with a sample per line in the format
'path of image start_frame verb_id noun_id'

@author: Γιώργος
"""

import os
import pickle
import cv2
import numpy as np
import torch.utils.data

def get_class_weights(list_file, num_classes, use_mapping):
    samples_list = parse_samples_list(list_file)
    counts = np.zeros(num_classes)
    mapping = None
    if use_mapping:
        mapping = make_class_mapping(samples_list)
        for s in samples_list:
            counts[mapping[s.label_verb]] += 1
    else:
        for s in samples_list:
            counts[s.label_verb] += 1
    
    weights = 1/counts
    weights = weights/np.sum(weights)
    return weights.astype(np.float32)

def make_class_mapping(samples_list):
    classes = []
    for sample in samples_list:
        if sample.label_verb not in classes:
            classes.append(sample.label_verb)
    classes = np.sort(classes)
    mapping_dict = {}
    for i, c in enumerate(classes):
        mapping_dict[c] = i
    return mapping_dict

def parse_samples_list(list_file):
    return [DataLine(x.strip().split(' ')) for x in open(list_file)]

def load_pickle(tracks_path):
    with open(tracks_path,'rb') as f:
        tracks = pickle.load(f)
    return tracks

class DataLine(object):
    def __init__(self, row):
        self.data = row

    @property
    def data_path(self):
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

class ImageDatasetLoader(torch.utils.data.Dataset):

    def __init__(self, list_file, num_classes=120,
                 batch_transform=None, channels='RGB', validation=False ):
        self.samples_list = parse_samples_list(list_file)
        if num_classes != 120:
            self.mapping = make_class_mapping(self.samples_list)
        else:
            self.mapping = None
        self.transform = batch_transform
        self.channels = channels
        self.validation = validation
        self.image_read_type = cv2.IMREAD_COLOR if channels=='RGB' else cv2.IMREAD_GRAYSCALE

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        img = cv2.imread(self.samples_list[index].data_path, self.image_read_type).astype(np.float32)
        if self.channels=='RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.mapping:
            class_id = self.mapping[self.samples_list[index].label_verb]
        else:
            class_id = self.samples_list[index].label_verb

        if not self.validation:
            return img, class_id
        else:
            name_parts = self.samples_list[index].data_path.split("\\")
            return img, class_id, name_parts[-2] + "\\" + name_parts[-1]

class VideoDatasetLoader(torch.utils.data.Dataset):

    def __init__(self, sampler, list_file, num_classes = 120, 
                 img_tmpl='img_{:05d}.jpg', batch_transform=None, validation=False):
        self.sampler = sampler
        self.video_list = parse_samples_list(list_file)
        if num_classes != 120:
            self.mapping = make_class_mapping(self.samples_list)
        else:
            self.mapping = None
        self.image_tmpl = img_tmpl
        self.transform = batch_transform
        self.validation = validation

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        frame_count=self.video_list[index].num_frames
        sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=index)

        sampled_frames = self.load_images(index, sampled_idxs)

        clip_input = np.concatenate(sampled_frames, axis=2)

        if self.transform is not None:
            clip_input = self.transform(clip_input)
            
        if self.mapping:
            class_id = self.mapping[self.video_list[index].label_verb]
        else:
            class_id = self.video_list[index].label_verb

        if not self.validation:
            return clip_input, class_id
        else:
            return clip_input, class_id, self.video_list[index].data_path.split("\\")[-1]

    def load_images(self, video_index, frame_indices):
        images = []
        for f_ind in frame_indices:
            im_name = os.path.join(self.video_list[video_index].data_path,
                                   self.image_tmpl.format(f_ind))
            next_image = cv2.imread(im_name, cv2.IMREAD_COLOR)
            next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
            images.append(next_image)
        return images

class PointDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, list_file, max_seq_length=None, num_classes=120, batch_transform=None, 
                 norm_val=[1.,1.,1.,1.], dual=False, clamp=False, validation=False):
        self.samples_list = parse_samples_list(list_file)
        if num_classes != 120:
            self.mapping = make_class_mapping(self.samples_list)
        else:
            self.mapping = None
        self.transform = batch_transform
        self.norm_val = np.array(norm_val)
        self.validation = validation
        self.max_seq_length = max_seq_length
        self.clamp = clamp
        
        self.data_arr = [load_pickle(self.samples_list[index].data_path) for index in range(len(self.samples_list))]
          
    def __len__(self):
        return len(self.samples_list)
    
    def __getitem__(self, index):
#        hand_tracks = load_pickle(self.samples_list[index].data_path)
        hand_tracks = self.data_arr[index]
        
        left_track = np.array(hand_tracks['left'], dtype=np.float32)
        left_track /= self.norm_val[:2]
        right_track = np.array(hand_tracks['right'], dtype=np.float32)
        right_track /= self.norm_val[2:]
        if self.clamp: # create new sequences with no zero points
            inds = np.where(left_track[:, 1] < 1.)
            if len(inds[0]) > 0: # in the extreme case where the hand is never in the segment we cannot clamp
                left_track = left_track[inds]
            inds = np.where(right_track[:, 1] < 1.)
            if len(inds[0]) > 0: 
                right_track = right_track[inds]
         
        if self.max_seq_length != 0: # indirectly supporting clamp without dual but will avoid experiments because it doesn't make much sense to combine the hand motions at different time steps
            left_track = left_track[np.linspace(0, len(left_track), self.max_seq_length, endpoint=False, dtype=int)]
            right_track = right_track[np.linspace(0, len(right_track), self.max_seq_length, endpoint=False, dtype=int)]
        
        points = np.concatenate((left_track, right_track), -1)
        seq_size = len(points)
        
        if self.mapping:
            class_id = self.mapping[self.samples_list[index].label_verb]
        else:
            class_id = self.samples_list[index].label_verb
        
        if not self.validation:
            return points, seq_size, class_id
        else:
            name_parts = self.samples_list[index].data_path.split("\\")
            return points, seq_size, class_id, name_parts[-2] + "\\" + name_parts[-1]

class PointVectorSummedDatasetLoader(torch.utils.data.Dataset):
    def __init__(self, list_file, max_seq_length=None, num_classes=120, 
                 dual=False, validation=False):
        self.samples_list = parse_samples_list(list_file)
        if num_classes != 120:
            self.mapping = make_class_mapping(self.samples_list)
        else:
            self.mapping = None
        self.validation = validation
        self.max_seq_length = max_seq_length
        self.dual = dual
        
        self.data_arr = [load_pickle(self.samples_list[index].data_path) for index in range(len(self.samples_list))]
        
    def __len__(self):
        return len(self.samples_list)
    
    def __getitem__(self, index):
#        hand_tracks = load_pickle(self.samples_list[index].data_path)
        hand_tracks = self.data_arr[index]
        left_track = np.array(hand_tracks['left'], dtype=np.int)
        right_track = np.array(hand_tracks['right'], dtype=np.int)   
        
        feat_size = 456+256
        feat_addon = 0
        if self.dual:
            feat_addon = feat_size
            feat_size *= 2
            
        vec = np.zeros((len(left_track), feat_size), dtype=np.float32)
        for i in range(len(left_track)):
            xl, yl = left_track[i]
            xr, yr = right_track[i]
            if yl < 256:
                vec[i:, xl] += 1
                vec[i:, 456+yl] += 1
            if yr < 256:
                vec[i:, feat_addon+xr] += 1
                vec[i:, feat_addon+456+yr] += 1
        
        if self.max_seq_length != 0:
            vec = vec[np.linspace(0, len(vec), self.max_seq_length, endpoint=False, dtype=int)]
            seq_size = self.max_seq_length
        else:
            seq_size = len(left_track)    
                
        if self.mapping:
            class_id = self.mapping[self.samples_list[index].label_verb]
        else:
            class_id = self.samples_list[index].label_verb

        if not self.validation:
            return vec, seq_size, class_id
        else:
            name_parts = self.samples_list[index].data_path.split("\\")
            return vec, seq_size, class_id, name_parts[-2] + "\\" + name_parts[-1]
                

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
        hand_tracks = load_pickle(self.samples_list[index].data_path)
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
    
    