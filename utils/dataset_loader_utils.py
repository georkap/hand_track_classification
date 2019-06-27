# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:42:50 2018

dataset loader utils such as:
    1) Custom transforms
    2) Custom collate functions (e.g. for lstm)

@author: Γιώργος
"""

import numpy as np
import cv2

from torch._six import string_classes, int_classes
import re
import collections
import torch

#### Custom image transforms ####

class Identity(object):
    def __call__(self, data):
        return data
    
class WidthCrop(object):
    def __call__(self, data):
        return data[:, 4:-4]

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    """
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.flipped = False

    def __call__(self, data):
        if self.rng.rand() < 0.5:
            data = np.fliplr(data)
            data = np.ascontiguousarray(data)
            self.flipped = True
        else:
            self.flipped = False
        return data

    def is_flipped(self):
        return self.flipped

class To01Range(object):
    def __init__(self, binarize):
        self.binarize = binarize
    
    def __call__(self, data):
        norm_image = cv2.normalize(data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)
        
        if len(norm_image.shape) == 2:
            norm_image = np.expand_dims(norm_image, -1)
        
        if self.binarize: 
            norm_image = np.where(norm_image > 0, 1., 0).astype(np.float32)
        
        return norm_image
        
class ResizePadFirst(object):
    def __init__(self, size, binarize, interpolation=cv2.INTER_LINEAR):
        self.size = size # int
        self.binarize = binarize
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
        
        if self.binarize: 
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

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class Resize(object):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """
    def __init__(self, size, binarize, interpolation=cv2.INTER_LINEAR):
        self.size = size # [w, h]
        self.binarize = binarize
        self.interpolation = interpolation
        self.new_shape = tuple()

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
            resize_blocks = c//512
            remaining = c%512
            if resize_blocks > 0:  # Even though it would fit, it will also do the batched resize if channels == 512, but I can avoid the check for channels= integer multiples of 512
                partial_resize = []
                for rbl in range(resize_blocks):
                    partial_scaled_data = cv2.resize(data[:,:,rbl*512:(rbl+1)*512], (new_w, new_h), interpolation=self.interpolation)
                    partial_resize.append(partial_scaled_data)
                if remaining:
                    partial_scaled_data = cv2.resize(data[:, :, resize_blocks*512:], (new_w, new_h), interpolation=self.interpolation)
                    partial_resize.append(partial_scaled_data)
                scaled_data = np.concatenate(partial_resize, axis=2)
            else:
                scaled_data = cv2.resize(data, (new_w, new_h), interpolation=self.interpolation)
        else:
            scaled_data = data

        if self.binarize: 
            scaled_data = np.where(scaled_data > 1, 255, 0).astype(np.float32)

        self.new_shape = scaled_data.shape

        return scaled_data

    def get_new_shape(self):
        return self.new_shape

class CenterCrop(object):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.tl = tuple()

    def __call__(self, data):
        h, w, c = data.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
        self.tl = (y1, x1)
        return cropped_data

    def get_tl(self):
        ''' Get top left (y, x) in image coordinates of the location where crop starts'''
        return self.tl

class RandomCrop(object):
    """Crops the given numpy array at the random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, seed=0):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.rng = np.random.RandomState(seed)
        self.tl = tuple()

    def __call__(self, data):
        h, w, c = data.shape
        th, tw = self.size
        x1 = self.rng.choice(range(w - tw))
        y1 = self.rng.choice(range(h - th))
        cropped_data = data[y1:(y1+th), x1:(x1+tw), :]
        self.tl = (y1, x1)
        return cropped_data

    def get_tl(self):
        ''' Get top left (y, x) in image coordinates of the location where crop starts'''
        return self.tl
    
class RandomScale(object):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """
    def __init__(self, make_square=False,
                       aspect_ratio=[1.0, 1.0],
                       slen=[224, 288],
                       interpolation=cv2.INTER_LINEAR, seed=0):
        assert slen[1] >= slen[0], \
                "slen ({}) should be in increase order".format(slen)
        assert aspect_ratio[1] >= aspect_ratio[0], \
                "aspect_ratio ({}) should be in increase order".format(aspect_ratio)
        self.slen = slen # [min factor, max factor]
        self.aspect_ratio = aspect_ratio
        self.make_square = make_square
        self.interpolation = interpolation
        self.rng = np.random.RandomState(seed)
        self.new_size = tuple()

    def __call__(self, data):
        h, w, c = data.shape
        new_w = w
        new_h = h if not self.make_square else w
        if self.aspect_ratio:
            random_aspect_ratio = self.rng.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
            if self.rng.rand() > 0.5:
                random_aspect_ratio = 1.0 / random_aspect_ratio
            new_w *= random_aspect_ratio
            new_h /= random_aspect_ratio
        resize_factor = self.rng.uniform(self.slen[0], self.slen[1]) / min(new_w, new_h)
        new_w *= resize_factor
        new_h *= resize_factor
        self.new_size = (int(new_w + 1), int(new_h + 1))
        scaled_data = cv2.resize(data, self.new_size, self.interpolation)
        return scaled_data

    def get_new_size(self):
        return self.new_size
    
class RandomHLS(object):
    def __init__(self, vars=[15, 35, 25]):
        self.vars = vars
        self.rng = np.random.RandomState(0)

    def __call__(self, data):
        h, w, c = data.shape
        assert c%3 == 0, "input channel = %d, illegal"%c

        random_vars = [int(round(self.rng.uniform(-x, x))) for x in self.vars]

        base = len(random_vars)
        augmented_data = np.zeros(data.shape, )

        for i_im in range(0, int(c/3)):
            augmented_data[:,:,3*i_im:(3*i_im+3)] = \
                    cv2.cvtColor(data[:,:,3*i_im:(3*i_im+3)], cv2.COLOR_RGB2HLS)

        hls_limits = [180, 255, 255]
        for ic in range(0, c):
            var = random_vars[ic%base]
            limit = hls_limits[ic%base]
            augmented_data[:,:,ic] = np.minimum(np.maximum(augmented_data[:,:,ic] + var, 0), limit)

        for i_im in range(0, int(c/3)):
            augmented_data[:,:,3*i_im:(3*i_im+3)] = \
                    cv2.cvtColor(augmented_data[:,:,3*i_im:(3*i_im+3)].astype(np.uint8), \
                        cv2.COLOR_HLS2RGB)

        return augmented_data

class ToTensorVid(object):
    """Converts a numpy.ndarray (H x W x (T x C)) in the range
    [0, 255] to a torch.FloatTensor of shape (C x T x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, clips):
        if isinstance(clips, np.ndarray):
            H, W, _ = clips.shape
            # handle numpy array
            clips = torch.from_numpy(clips.reshape((H,W,-1,self.dim)).transpose((3, 2, 0, 1)))
            # backward compatibility
            return clips.float() / 255.0
#### Custom collate functions

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

def lstm_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    
    if isinstance(batch[0], torch.Tensor):    
        _use_shared_memory = True

        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: lstm_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        # get the sequence lengths for all the samples of the batch
        seq_lengths = np.array([batch[i][1] for i in range(len(batch))])
        order_desc = np.argsort(seq_lengths)[::-1]
        max_seq_size = seq_lengths[order_desc][0]
        feature_size = len(batch[0][0][0])
#        num_times_feature = 1 if len(batch.shape)==3 else batch.shape[4]
        # sort the batch items descending according to the sequence lengths
        padded_batch = []
        for i, order in enumerate(order_desc):
#            if num_times_feature == 1:
            padded_inputs = np.zeros((max_seq_size, feature_size), dtype=np.float32)
#            else:
#                padded_inputs = np.zeros((1, max_seq_size, feature_size, num_times_feature), dtype=np.float32)
                
            padded_inputs[:batch[order][1], :] = batch[order][0]
                
            if len(batch[order]) == 4:
                appendable = (padded_inputs, batch[order][1], batch[order][2], batch[order][3])
            else:
                appendable = (padded_inputs, batch[order][1], batch[order][2])
            padded_batch.append(appendable)
        
        transposed = zip(*padded_batch)
        collated = []
        
        for samples in transposed:
            res = lstm_collate(samples)
            collated.append(res)
        return collated
        #return [my_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))    
    
    