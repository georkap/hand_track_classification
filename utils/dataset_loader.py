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
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import Dataset as torchDataset
from utils.video_sampler import RandomSampling, SequentialSampling, MiddleSampling


def get_class_weights(list_file, num_classes, use_mapping):
    samples_list = parse_samples_list(list_file, DataLine)
    counts = np.zeros(num_classes)
    mapping = None
    if use_mapping:
        mapping = make_class_mapping(samples_list)
        for s in samples_list:
            counts[mapping[s.label_verb]] += 1
    else:
        for s in samples_list:
            counts[s.label_verb] += 1

    weights = 1 / counts
    weights = weights / np.sum(weights)
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

def make_class_mapping_generic(samples_list, attribute):
    classes = []
    for sample in samples_list:
        label = getattr(sample, attribute)
        if label not in classes:
            classes.append(label)
    classes = np.sort(classes)
    mapping_dict = {}
    for i, c in enumerate(classes):
        mapping_dict[c] = i
    return mapping_dict


def load_pickle(tracks_path):
    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    return tracks


def substitute_prefix(tracks_path, secondary_prefix):
    obj_path = secondary_prefix
    for p in tracks_path.split('\\')[1:]:
        obj_path = os.path.join(obj_path, p)
    return obj_path

def load_two_pickle(tracks_path, secondary_prefix):
    obj_path = substitute_prefix(tracks_path, secondary_prefix)
    return load_pickle(tracks_path), load_pickle(obj_path)


def load_point_samples(samples_list, bpv_prefix=None):
    if bpv_prefix:
        data_arr = [load_two_pickle(samples_list[index].data_path, bpv_prefix) for index in range(len(samples_list))]
    else:
        data_arr = [load_pickle(samples_list[index].data_path) for index in range(len(samples_list))]
    return data_arr

# from PIL import Image
def load_images(data_path, frame_indices, image_tmpl):
    images = []
    for f_ind in frame_indices:
        im_name = os.path.join(data_path, image_tmpl.format(f_ind))
        # next_image = np.array(Image.open(im_name).convert('RGB'))
        next_image = cv2.imread(im_name, cv2.IMREAD_COLOR)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
        images.append(next_image)
    return images


def prepare_sampler(sampler_type, clip_length, frame_interval):
    if sampler_type == "train":
        train_sampler = RandomSampling(num=clip_length,
                                       interval=frame_interval,
                                       speed=[0.5, 1.5], seed=None)
        out_sampler = train_sampler
    else:
        val_sampler = SequentialSampling(num=clip_length,
                                         interval=frame_interval,
                                         fix_cursor=True,
                                         shuffle=True, seed=None)
        out_sampler = val_sampler
    return out_sampler


def object_list_to_bpv(detections, num_noun_classes, max_seq_length):
    sampled_detections = np.array(detections)
    if max_seq_length != 0:
        sampled_detections = sampled_detections[
            np.linspace(0, len(detections), max_seq_length, endpoint=False, dtype=int)].tolist()
        seq_length = max_seq_length
    else:
        seq_length = len(detections)
    bpv = np.zeros((seq_length, num_noun_classes), dtype=np.float32)
    for i, dets in enumerate(sampled_detections):
        for obj in dets:
            bpv[i, obj] = 1
    return bpv


def load_left_right_tracks(hand_tracks, max_seq_length):
    left_track = np.array(hand_tracks['left'], dtype=np.float32)
    right_track = np.array(hand_tracks['right'], dtype=np.float32)
    if max_seq_length != 0:
        left_track = left_track[np.linspace(0, len(left_track), max_seq_length, endpoint=False, dtype=int)]
        right_track = right_track[np.linspace(0, len(right_track), max_seq_length, endpoint=False, dtype=int)]
    return left_track, right_track


def calc_distance_differences(track):
    x2 = track[:, 0]
    x1 = np.roll(x2, 1)
    x1[0] = x1[1]
    y2 = track[:, 1]
    y1 = np.roll(y2, 1)
    y1[0] = y1[1]
    xdifs = x2 - x1
    ydifs = y2 - y1
    return np.concatenate((xdifs[:, np.newaxis], ydifs[:, np.newaxis]), -1)


def calc_angles(track):
    x2 = track[:, 0]
    x1 = np.roll(x2, 1)
    x1[0] = x1[1]
    y2 = track[:, 1]
    y1 = np.roll(y2, 1)
    y1[0] = y1[1]
    angles = np.arctan2(y2 * x1 - y1 * x2, x2 * x1 + y2 * y1, dtype=np.float32)
    return angles


def calc_polar_distance_from_prev(track):
    return np.concatenate((np.array([0]),
                           np.diagonal(squareform(pdist(track)), offset=-1)))


class DataLine(object):
    def __init__(self, row):
        self.data = row

    @property
    def data_path(self):
        return self.data[0]

    @property
    def num_frames(self):  # sto palio format ayto einai to start_frame
        return int(self.data[1])

    @property
    def label_verb(self):
        return int(self.data[2])

    @property
    def label_noun(self):
        return int(self.data[3])

    @property
    def uid(self):
        return int(self.data[4] if len(self.data) > 4 else -1)

    @property
    def start_frame(self):
        return int(self.data[5] if len(self.data) == 6 else -1)


class GTEADataLine(object):
    def __init__(self, row):
        self.data = row
        self.data_len = len(row)

    def get_video_path(self, prefix): # only used for FromVideoDatasetLoader and is deprecated
        return os.path.join(prefix, self.id_recipe, self.data_path + '.mp4')

    @property
    def data_path(self):
        return self.data[0]

    @property
    def frames_path(self):
        path_parts = os.path.normpath(self.data[0]).split(os.sep)
        session_parts = path_parts[1].split('-')
        session = session_parts[0] + '-' + session_parts[1] + '-' + session_parts[2]
        return os.path.join(path_parts[0], session, path_parts[1])

    @property
    def instance_name(self):
        return os.path.normpath(self.data[0]).split(os.sep)[1]

    @property
    def id_recipe(self):
        name_parts = self.data[0].split('-')
        id_recipe = name_parts[0] + '-' + name_parts[1] + '-' + name_parts[2]
        return id_recipe

    @property
    def label_action(self): # to zero based labels
        return int(self.data[1]) - 1 

    @property
    def label_verb(self):
        return int(self.data[2]) - 1 

    @property
    def label_noun(self):
        return int(self.data[3]) - 1

    @property
    def extra_nouns(self):
        extra_nouns = list()
        if self.data_len > 4:
            for noun in self.data[4:]:
                extra_nouns.append(int(noun) - 1)
        return extra_nouns


def parse_samples_list(list_file, datatype):
    return [datatype(x.strip().split(' ')) for x in open(list_file)]


class ImageDatasetLoader(torchDataset):

    def __init__(self, list_file, num_classes=120,
                 batch_transform=None, channels='RGB', validation=False):
        self.samples_list = parse_samples_list(list_file, DataLine)
        if num_classes != 120:
            self.mapping = make_class_mapping(self.samples_list)
        else:
            self.mapping = None
        self.transform = batch_transform
        self.channels = channels
        self.validation = validation
        self.image_read_type = cv2.IMREAD_COLOR if channels == 'RGB' else cv2.IMREAD_GRAYSCALE

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        img = cv2.imread(self.samples_list[index].data_path, self.image_read_type).astype(np.float32)
        if self.channels == 'RGB':
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


class Video(object):
    # adapted from https://github.com/cypw/PyTorch-MFNet/blob/master/data/video_iterator.py
    """basic Video class"""

    def __init__(self, vid_path):
        self.open(vid_path)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def reset(self):
        self.close()
        self.vid_path = None
        self.frame_count = -1
        self.faulty_frame = None
        return self

    def open(self, vid_path):
        assert os.path.exists(vid_path), "VideoIter:: cannot locate: `{}'".format(vid_path)

        # close previous video & reset variables
        self.reset()

        # try to open video
        cap = cv2.VideoCapture(vid_path)
        if cap.isOpened():
            self.cap = cap
            self.vid_path = vid_path
        else:
            raise IOError("VideoIter:: failed to open video: `{}'".format(vid_path))

        return self

    def count_frames(self, check_validity=False):
        offset = 0
        if self.vid_path.endswith('.flv'):
            offset = -1
        unverified_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) + offset
        if check_validity:
            verified_frame_count = 0
            for i in range(unverified_frame_count):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                if not self.cap.grab():
                    print("VideoIter:: >> frame (start from 0) {} corrupted in {}".format(i, self.vid_path))
                    break
                verified_frame_count = i + 1
            self.frame_count = verified_frame_count
        else:
            self.frame_count = unverified_frame_count
        assert self.frame_count > 0, "VideoIter:: Video: `{}' has no frames".format(self.vid_path)
        return self.frame_count

    def extract_frames(self, idxs, force_color=True):
        frames = self.extract_frames_fast(idxs, force_color)
        if frames is None:
            # try slow method:
            frames = self.extract_frames_slow(idxs, force_color)
        return frames

    def extract_frames_fast(self, idxs, force_color=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = []
        pre_idx = max(idxs)
        for idx in idxs:
            assert (self.frame_count < 0) or (idx < self.frame_count), \
                "idxs: {} > total valid frames({})".format(idxs, self.frame_count)
            if pre_idx != (idx - 1):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, frame = self.cap.read() # in BGR/GRAY format
            pre_idx = idx
            if not res:
                self.faulty_frame = idx
                return None
            if len(frame.shape) < 3:
                if force_color:
                    # Convert Gray to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames

    def extract_frames_slow(self, idxs, force_color=True):
        assert self.cap is not None, "No opened video."
        if len(idxs) < 1:
            return []

        frames = [None] * len(idxs)
        idx = min(idxs)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        while idx <= max(idxs):
            res, frame = self.cap.read() # in BGR/GRAY format
            if not res:
                # end of the video
                self.faulty_frame = idx
                return None
            if idx in idxs:
                # fond a frame
                if len(frame.shape) < 3:
                    if force_color:
                        # Convert Gray to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pos = [k for k, i in enumerate(idxs) if i == idx]
                for k in pos:
                    frames[k] = frame
            idx += 1
        return frames

    def close(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        return self

class VideoFromImagesDatasetLoader(torchDataset): # loads GTEA dataset from frames
    OBJECTIVE_NAMES = ['label_action', 'label_verb', 'label_noun']
    def __init__(self, sampler, split_file, line_type, num_classes, max_num_classes, img_tmpl='img_{:05d}.jpg',
                 batch_transform=None, extra_nouns=False, use_gaze=False, gaze_list_prefix=None, use_hands=False,
                 hand_list_prefix=None, validation=False, vis_data=False):
        self.sampler = sampler
        self.video_list = parse_samples_list(split_file, GTEADataLine)  # if line_type=='GTEA' else DataLine)
        self.extra_nouns = extra_nouns
        self.usable_objectives = list()
        self.mappings = list()
        for i, (objective, objective_name) in enumerate(zip(num_classes, FromVideoDatasetLoader.OBJECTIVE_NAMES)):
            self.usable_objectives.append(objective > 0)
            if objective != max_num_classes[i] and self.usable_objectives[-1]:
                self.mappings.append(make_class_mapping_generic(self.video_list, objective_name))
            else:
                self.mappings.append(None)
        assert any(obj is True for obj in self.usable_objectives)
        self.transform = batch_transform
        self.validation = validation
        self.vis_data = vis_data
        self.use_gaze = use_gaze
        self.gaze_list_prefix = gaze_list_prefix
        self.use_hands = use_hands
        self.hand_list_prefix = hand_list_prefix
        self.norm_val = [640., 480., 640., 480.]
        self.image_tmpl = img_tmpl

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        path = self.video_list[index].frames_path
        instance_name = self.video_list[index].instance_name
        frame_count = len(os.listdir(path))
        assert frame_count > 0

        sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=index, start_frame=0)

        sampled_frames = load_images(path, sampled_idxs, self.image_tmpl)

        clip_input = np.concatenate(sampled_frames, axis=2)

        gaze_points = None
        if self.use_gaze:
            pass
        hand_points = None
        if self.use_hands: # almost the same process as VideoAndPointDatasetLoader
            hand_track_path = os.path.join(self.hand_list_prefix, instance_name + '.pkl')
            hand_tracks = load_pickle(hand_track_path)

            left_track = np.array(hand_tracks['left'], dtype=np.float32)
            right_track = np.array(hand_tracks['right'], dtype=np.float32)

            left_track = left_track[sampled_idxs]  # keep the points for the sampled frames
            right_track = right_track[sampled_idxs]
            if not self.vis_data:
                left_track = left_track[::2]  # keep 1 coordinate pair for every two frames because we supervise 8 outputs from the temporal dim of mfnet and not 16 as the inputs
                right_track = right_track[::2]

            norm_val = self.norm_val
            if self.transform is not None:
                or_h, or_w, _ = clip_input.shape
                clip_input = self.transform(clip_input) # have to put this line here for compatibility with the hand transform code
                is_flipped = False
                if 'RandomScale' in self.transform.transforms[
                    0].__repr__():  # means we are in training so get the transformations
                    sc_w, sc_h = self.transform.transforms[0].get_new_size()
                    tl_y, tl_x = self.transform.transforms[1].get_tl()
                    if 'RandomHorizontalFlip' in self.transform.transforms[2].__repr__():
                        is_flipped = self.transform.transforms[2].is_flipped()
                elif 'Resize' in self.transform.transforms[0].__repr__():  # means we are in testing
                    sc_h, sc_w, _ = self.transform.transforms[0].get_new_shape()
                    tl_y, tl_x = self.transform.transforms[1].get_tl()
                else:
                    sc_w = or_w
                    sc_h = or_h
                    tl_x = 0
                    tl_y = 0

                # apply transforms to tracks
                scale_x = sc_w / or_w
                scale_y = sc_h / or_h
                left_track *= [scale_x, scale_y]
                left_track -= [tl_x, tl_y]
                right_track *= [scale_x, scale_y]
                right_track -= [tl_x, tl_y]

                _, _, max_h, max_w = clip_input.shape
                norm_val = [max_w, max_h, max_w, max_h]
                if is_flipped:
                    left_track[:, 0] = max_w - left_track[:, 0]
                    right_track[:, 0] = max_w - right_track[:, 0]

            if self.vis_data:
                left_track_vis = left_track
                right_track_vis = right_track
                left_track = left_track[::2]
                right_track = right_track[::2]
            # for the DSNT layer normalize to [-1, 1] for x and to [-1, 2] for y, which can get values greater than +1 when the hand is originally not detected
            left_track = (left_track * 2 + 1) / norm_val[:2] - 1
            right_track = (right_track * 2 + 1) / norm_val[2:] - 1
            hand_points = np.concatenate((left_track[:, np.newaxis, :], right_track[:, np.newaxis, :]), axis=1).astype(np.float32)
            hand_points = hand_points.flatten()

        # apply transforms on the video clip
        if self.transform is not None and not (self.use_hands or self.use_gaze):
            clip_input = self.transform(clip_input)

        # get the labels for the tasks
        labels = list()
        if self.usable_objectives[0]:
            action_id = self.video_list[index].label_action
            if self.mappings[0]:
                action_id = self.mappings[0][action_id]
            labels.append(action_id)
        if self.usable_objectives[1]:
            verb_id = self.video_list[index].label_verb
            if self.mappings[1]:
                verb_id = self.mappings[1][verb_id]
            labels.append(verb_id)
        if self.usable_objectives[2]:
            noun_id = self.video_list[index].label_noun
            if self.mappings[2]:
                noun_id = self.mappings[2][noun_id]
            labels.append(noun_id)

            if self.extra_nouns:
                extra_nouns = self.video_list[index].extra_nouns
                if self.mappings[2]:
                    extra_nouns = [self.mappings[2][en] for en in extra_nouns]
                for en in extra_nouns:
                    labels.append(en)

        if self.use_gaze or self.use_hands:
            labels = np.array(labels, dtype=np.float32)
        else:
            labels = np.array(labels, dtype=np.int64)  # numpy array for pytorch dataloader compatibility
        if self.use_gaze:
            labels = np.concatenate((labels, gaze_points))
        if self.use_hands:
            labels = np.concatenate((labels, hand_points))

        if self.vis_data:
            # for i in range(len(sampled_frames)):
            #     cv2.imshow('orig_img', sampled_frames[i])
            #     cv2.imshow('transform', clip_input[:, i, :, :].numpy().transpose(1, 2, 0))
            #     cv2.waitKey(0)

            def vis_with_circle(img, left_point, right_point, winname):
                k = cv2.circle(img.copy(), (int(left_point[0]), int(left_point[1])), 10, (255, 0, 0), 4)
                k = cv2.circle(k, (int(right_point[0]), int(right_point[1])), 10, (0, 0, 255), 4)
                cv2.imshow(winname, k)

            orig_left = np.array(hand_tracks['left'], dtype=np.float32)
            orig_left = orig_left[sampled_idxs]
            orig_right = np.array(hand_tracks['right'], dtype=np.float32)
            orig_right = orig_right[sampled_idxs]

            for i in range(len(sampled_frames)):
                vis_with_circle(sampled_frames[i], orig_left[i], orig_right[i], 'no augmentation')
                vis_with_circle(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), left_track_vis[i], right_track_vis[i],
                                'transformed')
                vis_with_circle(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), orig_left[i], orig_right[i],
                                'transf_img_not_coords')
                cv2.waitKey(0)

        if not self.validation:
            return clip_input, labels
        else:
            return clip_input, labels, instance_name


from gulpio import GulpDirectory
class FromVideoDatasetLoaderGulp(torchDataset): #loads GTEA dataset from gulp
    OBJECTIVE_NAMES = ['label_action', 'label_verb', 'label_noun']
    def __init__(self, sampler, split_file, line_type, num_classes, max_num_classes, batch_transform=None,
                 extra_nouns=False, use_gaze=False, gaze_list_prefix=None, use_hands=False, hand_list_prefix=None,
                 validation=False, vis_data=False):
        self.sampler = sampler
        self.video_list = parse_samples_list(split_file, GTEADataLine)  # if line_type=='GTEA' else DataLine)
        self.extra_nouns = extra_nouns
        self.usable_objectives = list()
        self.mappings = list()
        for i, (objective, objective_name) in enumerate(zip(num_classes, FromVideoDatasetLoader.OBJECTIVE_NAMES)):
            self.usable_objectives.append(objective > 0)
            if objective != max_num_classes[i] and self.usable_objectives[-1]:
                self.mappings.append(make_class_mapping_generic(self.video_list, objective_name))
            else:
                self.mappings.append(None)
        assert any(obj is True for obj in self.usable_objectives)
        self.transform = batch_transform
        self.validation = validation
        self.vis_data = vis_data
        self.use_gaze = use_gaze
        self.gaze_list_prefix = gaze_list_prefix
        self.use_hands = use_hands
        self.hand_list_prefix = hand_list_prefix
        self.norm_val = [640., 480., 640., 480.]

        # gulp_data_dir = r"D:\Datasets\egocentric\GTEA\gulp_output2"
        gulp_data_dir = r"F:\workspace_George\GTEA\gteagulp"
        self.gd = GulpDirectory(gulp_data_dir)
        # self.items = list(self.gd.merged_meta_dict.items())
        self.merged_data_dict = self.gd.merged_meta_dict
        self.num_chunks = self.gd.num_chunks
        self.data_path = gulp_data_dir

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        # item_id, item_info = self.items[index]
        # assert item_id == self.video_list[index].data_path
        path = self.video_list[index].data_path
        item_info = self.merged_data_dict[path]
        frame_count = len(item_info['frame_info'])
        assert frame_count > 0

        sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=index, start_frame=0)
        # sampled_idxs = [10,11,13,14,15,15,15,15]
        sampler_step = self.sampler.interval
        produced_step = np.mean(sampled_idxs[1:] - np.roll(sampled_idxs,1)[1:])
        if sampler_step[0] == produced_step:
            sampled_frames, meta = self.gd[path, slice(sampled_idxs[0], sampled_idxs[-1]+1, sampler_step[0])]
        else:
            imgs, meta = self.gd[path]
            assert sampled_idxs[-1] < len(imgs)
            sampled_frames = []
            for i in sampled_idxs:
                sampled_frames.append(imgs[i])

        clip_input = np.concatenate(sampled_frames, axis=2)

        gaze_points = None
        if self.use_gaze:
            pass
        hand_points = None
        if self.use_hands: # almost the same process as VideoAndPointDatasetLoader
            hand_track_path = os.path.join(self.hand_list_prefix, path + '.pkl')
            hand_tracks = load_pickle(hand_track_path)

            left_track = np.array(hand_tracks['left'], dtype=np.float32)
            right_track = np.array(hand_tracks['right'], dtype=np.float32)

            left_track = left_track[sampled_idxs]  # keep the points for the sampled frames
            right_track = right_track[sampled_idxs]
            if not self.vis_data:
                left_track = left_track[::2]  # keep 1 coordinate pair for every two frames because we supervise 8 outputs from the temporal dim of mfnet and not 16 as the inputs
                right_track = right_track[::2]

            norm_val = self.norm_val
            if self.transform is not None:
                or_h, or_w, _ = clip_input.shape
                clip_input = self.transform(clip_input) # have to put this line here for compatibility with the hand transform code
                is_flipped = False
                if 'RandomScale' in self.transform.transforms[
                    0].__repr__():  # means we are in training so get the transformations
                    sc_w, sc_h = self.transform.transforms[0].get_new_size()
                    tl_y, tl_x = self.transform.transforms[1].get_tl()
                    if 'RandomHorizontalFlip' in self.transform.transforms[2].__repr__():
                        is_flipped = self.transform.transforms[2].is_flipped()
                elif 'Resize' in self.transform.transforms[0].__repr__():  # means we are in testing
                    sc_h, sc_w, _ = self.transform.transforms[0].get_new_shape()
                    tl_y, tl_x = self.transform.transforms[1].get_tl()
                else:
                    sc_w = or_w
                    sc_h = or_h
                    tl_x = 0
                    tl_y = 0

                # apply transforms to tracks
                scale_x = sc_w / or_w
                scale_y = sc_h / or_h
                left_track *= [scale_x, scale_y]
                left_track -= [tl_x, tl_y]
                right_track *= [scale_x, scale_y]
                right_track -= [tl_x, tl_y]

                _, _, max_h, max_w = clip_input.shape
                norm_val = [max_w, max_h, max_w, max_h]
                if is_flipped:
                    left_track[:, 0] = max_w - left_track[:, 0]
                    right_track[:, 0] = max_w - right_track[:, 0]

            if self.vis_data:
                left_track_vis = left_track
                right_track_vis = right_track
                left_track = left_track[::2]
                right_track = right_track[::2]
            # for the DSNT layer normalize to [-1, 1] for x and to [-1, 2] for y, which can get values greater than +1 when the hand is originally not detected
            left_track = (left_track * 2 + 1) / norm_val[:2] - 1
            right_track = (right_track * 2 + 1) / norm_val[2:] - 1
            hand_points = np.concatenate((left_track[:, np.newaxis, :], right_track[:, np.newaxis, :]), axis=1).astype(np.float32)
            hand_points = hand_points.flatten()

        # apply transforms on the video clip
        if self.transform is not None and not (self.use_hands or self.use_gaze):
            clip_input = self.transform(clip_input)

        # get the labels for the tasks
        labels = list()
        if self.usable_objectives[0]:
            action_id = self.video_list[index].label_action
            if self.mappings[0]:
                action_id = self.mappings[0][action_id]
            labels.append(action_id)
        if self.usable_objectives[1]:
            verb_id = self.video_list[index].label_verb
            if self.mappings[1]:
                verb_id = self.mappings[1][verb_id]
            labels.append(verb_id)
        if self.usable_objectives[2]:
            noun_id = self.video_list[index].label_noun
            if self.mappings[2]:
                noun_id = self.mappings[2][noun_id]
            labels.append(noun_id)

            if self.extra_nouns:
                extra_nouns = self.video_list[index].extra_nouns
                if self.mappings[2]:
                    extra_nouns = [self.mappings[2][en] for en in extra_nouns]
                for en in extra_nouns:
                    labels.append(en)

        if self.use_gaze or self.use_hands:
            labels = np.array(labels, dtype=np.float32)
        else:
            labels = np.array(labels, dtype=np.int64)  # numpy array for pytorch dataloader compatibility
        if self.use_gaze:
            labels = np.concatenate((labels, gaze_points))
        if self.use_hands:
            labels = np.concatenate((labels, hand_points))

        if self.vis_data:
            # for i in range(len(sampled_frames)):
            #     cv2.imshow('orig_img', sampled_frames[i])
            #     cv2.imshow('transform', clip_input[:, i, :, :].numpy().transpose(1, 2, 0))
            #     cv2.waitKey(0)

            def vis_with_circle(img, left_point, right_point, winname):
                k = cv2.circle(img.copy(), (int(left_point[0]), int(left_point[1])), 10, (255, 0, 0), 4)
                k = cv2.circle(k, (int(right_point[0]), int(right_point[1])), 10, (0, 0, 255), 4)
                cv2.imshow(winname, k)

            orig_left = np.array(hand_tracks['left'], dtype=np.float32)
            orig_left = orig_left[sampled_idxs]
            orig_right = np.array(hand_tracks['right'], dtype=np.float32)
            orig_right = orig_right[sampled_idxs]

            for i in range(len(sampled_frames)):
                vis_with_circle(sampled_frames[i], orig_left[i], orig_right[i], 'no augmentation')
                vis_with_circle(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), left_track_vis[i], right_track_vis[i],
                                'transformed')
                vis_with_circle(clip_input[:, i, :, :].numpy().transpose(1, 2, 0), orig_left[i], orig_right[i],
                                'transf_img_not_coords')
                cv2.waitKey(0)

        if not self.validation:
            return clip_input, labels
        else:
            return clip_input, labels, self.video_list[index].data_path


class FromVideoDatasetLoader(torchDataset): # loads gtea dataset from video files; not gonna be using anymore
    OBJECTIVE_NAMES = ['label_action', 'label_verb', 'label_noun']
    def __init__(self, sampler, split_file, line_type, num_classes, max_num_classes, batch_transform=None, extra_nouns=False,
                 validation=False, vis_data=False):
        self.sampler = sampler
        self.video_list = parse_samples_list(split_file, GTEADataLine)  # if line_type=='GTEA' else DataLine)
        self.extra_nouns = extra_nouns

        # num_classes is a list with 3 integers.
        # num_classes[0] = num_actions,
        # num_classes[1] = num_verbs,
        # num_classes[2] = num_nouns
        # if any of these has the value <= 0 then this objective will not be used in the network
        # if any of these has value different than its respective on max_num_classes then I perform class mapping
        # max_num_classes is a list with 3 integers which define the maximum number of classes for the objective and is
        # fixed for certain dataset. E.g. for EPIC it is [0, 125, 322], for GTEA it is [106, 19, 53]
        self.usable_objectives = list()
        self.mappings = list()
        for i, (objective, objective_name) in enumerate(zip(num_classes, FromVideoDatasetLoader.OBJECTIVE_NAMES)):
            self.usable_objectives.append(objective > 0)
            if objective != max_num_classes[i] and self.usable_objectives[-1]:
                self.mappings.append(make_class_mapping_generic(self.video_list, objective_name))
            else:
                self.mappings.append(None)
        assert any(obj is True for obj in self.usable_objectives)
        self.transform = batch_transform
        self.validation = validation
        self.vis_data = vis_data

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        sampled_frames = []
        try:
            with Video(vid_path=self.video_list[index].get_video_path(prefix='gtea_clips')) as vid:
                start_frame = 0
                frame_count = vid.count_frames(check_validity=False)
                sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=index, start_frame=start_frame)
                sampled_frames = vid.extract_frames(idxs=sampled_idxs, force_color=True)
        except IOError as e:
            print(">> I/O error({0}): {1}".format(e.errno, e.strerror))

        clip_input = np.concatenate(sampled_frames, axis=2)

        if self.transform is not None:
            clip_input = self.transform(clip_input)

        labels = list()
        if self.usable_objectives[0]:
            action_id = self.video_list[index].label_action
            if self.mappings[0]:
                action_id = self.mappings[0][action_id]
            labels.append(action_id)
        if self.usable_objectives[1]:
            verb_id = self.video_list[index].label_verb
            if self.mappings[1]:
                verb_id = self.mappings[1][verb_id]
            labels.append(verb_id)
        if self.usable_objectives[2]:
            noun_id = self.video_list[index].label_noun
            if self.mappings[2]:
                noun_id = self.mappings[2][noun_id]
            labels.append(noun_id)

            if self.extra_nouns:
                extra_nouns = self.video_list[index].extra_nouns
                if self.mappings[2]:
                    extra_nouns = [self.mappings[2][en] for en in extra_nouns]
                for en in extra_nouns:
                    labels.append(en)

        labels = np.array(labels, dtype=np.int64) # for pytorch dataloader compatibility

        if self.vis_data:
            for i in range(len(sampled_frames)):
                cv2.imshow('orig_img', sampled_frames[i])
                cv2.imshow('transform', clip_input[:,i,:,:].numpy().transpose(1,2,0))
                cv2.waitKey(0)

        if not self.validation:
            return clip_input, labels
        else:
            return clip_input, labels, self.video_list[index].data_path



class VideoDatasetLoader(torchDataset):

    def __init__(self, sampler, list_file, num_classes=120,
                 img_tmpl='img_{:05d}.jpg', batch_transform=None, validation=False):
        self.sampler = sampler
        self.video_list = parse_samples_list(list_file, DataLine)

        # check for double output and choose as first the verb classes
        if not isinstance(num_classes, tuple):
            verb_classes = num_classes
        else:
            verb_classes = num_classes[0]

        if verb_classes != 120:
            self.mapping = make_class_mapping(self.video_list)
        else:
            self.mapping = None
        self.double_output = isinstance(num_classes, tuple)
        self.image_tmpl = img_tmpl
        self.transform = batch_transform
        self.validation = validation

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        frame_count = self.video_list[index].num_frames
        start_frame = self.video_list[index].start_frame
        start_frame = start_frame if start_frame != -1 else 0
        sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=index,
                                             start_frame=start_frame)

        sampled_frames = load_images(self.video_list[index].data_path, sampled_idxs, self.image_tmpl)

        clip_input = np.concatenate(sampled_frames, axis=2)

        if self.transform is not None:
            clip_input = self.transform(clip_input)

        if self.mapping:
            verb_id = self.mapping[self.video_list[index].label_verb]
        else:
            verb_id = self.video_list[index].label_verb
        if self.double_output:
            noun_id = self.video_list[index].label_noun
            classes = (verb_id, noun_id) # np.array([verb_id, noun_id], dtype=np.int64) should refactor to this for double output
        else:
            classes = verb_id

        if not self.validation:
            return clip_input, classes
        else:
            return clip_input, classes, self.video_list[index].uid #self.video_list[index].data_path.split("\\")[-1]


# TODO: this is for sliding window sample creation with a fixed sizes
class PointPolarDatasetLoaderMultiSec(torchDataset):
    def __init__(self, list_file, max_seq_length=None, norm_val=None,
                 validation=False):
        self.samples_list = parse_samples_list(list_file, DataLine)
        # no mapping supported for now. only use all classes
        self.norm_val = np.array(norm_val)
        self.max_seq_length = max_seq_length
        self.validation = validation
        self.data_arr = []
        for index in range(len(self.samples_list)):
            hand_tracks = load_pickle(self.samples_list[index].data_path)
            left_track = np.array(hand_tracks['left'], dtype=np.float32)
            right_track = np.array(hand_tracks['right'], dtype=np.float32)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class PointDiffDatasetLoader(torchDataset):
    def __init__(self, list_file, max_seq_length=None, norm_val=None,
                 validation=False):
        self.samples_list = parse_samples_list(list_file, DataLine)
        self.norm_val = np.array(norm_val)
        self.max_seq_length = max_seq_length
        self.validation = validation
        self.data_arr = [load_pickle(self.samples_list[index].data_path) for index in range(len(self.samples_list))]

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        left_track, right_track = load_left_right_tracks(self.data_arr[index], self.max_seq_length)

        left_track /= self.norm_val[:2]
        right_track /= self.norm_val[2:]

        left_diffs = calc_distance_differences(left_track)
        right_diffs = calc_distance_differences(right_track)

        points = np.concatenate((left_track, left_diffs, right_track, right_diffs), -1).astype(np.float32)
        seq_size = len(points)
        class_id = self.samples_list[index].label_verb
        if not self.validation:
            return points, seq_size, class_id
        else:
            name_parts = self.samples_list[index].data_path.split("\\")
            return points, seq_size, class_id, name_parts[-2] + "\\" + name_parts[-1]


class AnglesDatasetLoader(torchDataset):
    def __init__(self, list_file, max_seq_length=None, validation=False):
        self.samples_list = parse_samples_list(list_file, DataLine)
        # no mapping supported for now. only use all classes
        self.max_seq_length = max_seq_length
        self.validation = validation
        self.data_arr = [load_pickle(self.samples_list[index].data_path) for index in range(len(self.samples_list))]

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        left_track, right_track = load_left_right_tracks(self.data_arr[index], self.max_seq_length)

        left_angles = calc_angles(left_track)
        right_angles = calc_angles(right_track)

        points = np.concatenate((left_angles[:, np.newaxis],
                                 right_angles[:, np.newaxis]), -1).astype(np.float32)
        seq_size = len(points)

        class_id = self.samples_list[index].label_verb
        if not self.validation:
            return points, seq_size, class_id
        else:
            name_parts = self.samples_list[index].data_path.split("\\")
            return points, seq_size, class_id, name_parts[-2] + "\\" + name_parts[-1]


class PointPolarDatasetLoader(torchDataset):

    def __init__(self, list_file, max_seq_length=None, norm_val=None,
                 validation=False):
        self.samples_list = parse_samples_list(list_file, DataLine)
        # no mapping supported for now. only use all classes
        self.norm_val = np.array(norm_val)
        self.max_seq_length = max_seq_length
        self.validation = validation
        self.data_arr = [load_pickle(self.samples_list[index].data_path) for index in range(len(self.samples_list))]

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        left_track, right_track = load_left_right_tracks(self.data_arr[index], self.max_seq_length)

        left_angles = calc_angles(left_track)
        right_angles = calc_angles(right_track)

        left_track /= self.norm_val[:2]
        right_track /= self.norm_val[2:]

        left_dist = np.concatenate((np.array([0]),
                                    np.diagonal(squareform(pdist(left_track)), offset=-1)))
        right_dist = np.concatenate((np.array([0]),
                                     np.diagonal(squareform(pdist(right_track)), offset=-1)))

        points = np.concatenate((left_track,
                                 left_dist[:, np.newaxis],
                                 left_angles[:, np.newaxis],
                                 right_track,
                                 right_dist[:, np.newaxis],
                                 right_angles[:, np.newaxis]), -1).astype(np.float32)
        seq_size = len(points)

        class_id = self.samples_list[index].label_verb
        if not self.validation:
            return points, seq_size, class_id
        else:
            name_parts = self.samples_list[index].data_path.split("\\")
            return points, seq_size, class_id, name_parts[-2] + "\\" + name_parts[-1]


class PointObjDatasetLoader(torchDataset):
    def __init__(self, list_file, max_seq_length, double_output,
                 norm_val=None, bpv_prefix='noun_bpv_oh', validation=False):
        self.samples_list = parse_samples_list(list_file, DataLine)
        # no mapping supported for now. only use all classes
        self.norm_val = np.array(norm_val)
        self.validation = validation
        self.double_output = double_output
        self.max_seq_length = max_seq_length
        self.data_arr = [load_two_pickle(self.samples_list[index].data_path, bpv_prefix) for index in
                         range(len(self.samples_list))]

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        hand_tracks, object_tracks = self.data_arr[index]
        left_track, right_track = load_left_right_tracks(hand_tracks, self.max_seq_length)

        left_track /= self.norm_val[:2]
        right_track /= self.norm_val[2:]

        if self.max_seq_length != 0:
            object_tracks = object_tracks[
                np.linspace(0, len(object_tracks), self.max_seq_length, endpoint=False, dtype=int)]
        object_tracks = object_tracks / np.tile(self.norm_val[:2], 352)

        points = np.concatenate((left_track, right_track, object_tracks),
                                -1).astype(np.float32)
        seq_size = len(points)

        verb_id = self.samples_list[index].label_verb
        if self.double_output:
            noun_id = self.samples_list[index].label_noun
            classes = np.array([verb_id, noun_id], dtype=np.int64)
        else:
            classes = verb_id

        if not self.validation:
            return points, seq_size, classes
        else:
            name_parts = self.samples_list[index].data_path.split("\\")
            return points, seq_size, classes, name_parts[-2] + "\\" + name_parts[-1]


class PointBpvDatasetLoader(torchDataset):
    def __init__(self, list_file, max_seq_length, double_output,
                 norm_val=None, bpv_prefix='noun_bpv_oh', validation=False, num_workers=0):
        self.samples_list = parse_samples_list(list_file, DataLine)
        # no mapping supported for now. only use all classes
        self.norm_val = np.array(norm_val)
        self.validation = validation
        self.double_output = double_output
        self.max_seq_length = max_seq_length
        self.bpv_prefix = bpv_prefix

        #        if not data_arr:
        #            self.data_arr = make_data_arr(self.samples_list, bpv_prefix)
        #        else:
        #            self.data_arr = data_arr
        if num_workers == 0:
            self.data_arr = load_point_samples(self.samples_list, bpv_prefix)
        else:
            self.data_arr = None

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        if self.data_arr is not None:
            hand_tracks, object_detections = self.data_arr[index]
        else:
            hand_tracks, object_detections = load_two_pickle(self.samples_list[index].data_path,
                                                             self.bpv_prefix)
        left_track, right_track = load_left_right_tracks(hand_tracks, self.max_seq_length)

        left_track /= self.norm_val[:2]
        right_track /= self.norm_val[2:]

        bpv = object_list_to_bpv(object_detections, 352, self.max_seq_length)

        points = np.concatenate((left_track,
                                 right_track,
                                 bpv), -1).astype(np.float32)
        seq_size = len(points)

        verb_id = self.samples_list[index].label_verb
        if self.double_output:
            noun_id = self.samples_list[index].label_noun
            classes = np.array([verb_id, noun_id], dtype=np.int64)
        else:
            classes = verb_id

        if not self.validation:
            return points, seq_size, classes
        else:
            name_parts = self.samples_list[index].data_path.split("\\")
            return points, seq_size, classes, name_parts[-2] + "\\" + name_parts[-1]


class PointDatasetLoader(torchDataset):
    def __init__(self, list_file, max_seq_length=None, num_classes=120,
                 batch_transform=None, norm_val=None, dual=False,
                 clamp=False, only_left=False, only_right=False, validation=False):
        self.samples_list = parse_samples_list(list_file, DataLine)
        if num_classes != 120 and num_classes != 125:  # TODO: find a better way to apply mapping
            self.mapping = make_class_mapping(self.samples_list)
        else:
            self.mapping = None
        self.transform = batch_transform
        self.norm_val = np.array(norm_val)
        self.validation = validation
        self.max_seq_length = max_seq_length
        self.clamp = clamp
        self.only_left = only_left
        self.only_right = only_right

        self.data_arr = load_point_samples(self.samples_list, None)

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        #        hand_tracks = load_pickle(self.samples_list[index].data_path)
        hand_tracks: dict = self.data_arr[index]

        left_track = np.array(hand_tracks['left'], dtype=np.float32)
        left_track /= self.norm_val[:2]
        right_track = np.array(hand_tracks['right'], dtype=np.float32)
        right_track /= self.norm_val[2:]
        if self.clamp:  # create new sequences with no zero points
            inds = np.where(left_track[:, 1] < 1.)
            if len(inds[0]) > 0:  # in the extreme case where the hand is never in the segment we cannot clamp
                left_track = left_track[inds]
            inds = np.where(right_track[:, 1] < 1.)
            if len(inds[0]) > 0:
                right_track = right_track[inds]

        if self.max_seq_length != 0:  # indirectly supporting clamp without dual but will avoid experiments because it doesn't make much sense to combine the hand motions at different time steps
            left_track = left_track[np.linspace(0, len(left_track), self.max_seq_length, endpoint=False, dtype=int)]
            right_track = right_track[np.linspace(0, len(right_track), self.max_seq_length, endpoint=False, dtype=int)]

        if self.only_left:
            points = left_track
        elif self.only_right:
            points = right_track
        else:
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


class VideoAndPointDatasetLoader(torchDataset):
    EPIC_MAX_CLASSES = [0, 125, 322]
    def __init__(self, sampler, video_list_file, point_list_prefix, num_classes, img_tmpl='img_{:05d}.jpg', # removed predefined argument from num_classes
                 norm_val=None, batch_transform=None, validation=False, vis_data=False):
        self.sampler = sampler
        self.video_list = parse_samples_list(video_list_file, DataLine)

        if isinstance(num_classes, int): # old workflow for backwards compatibility to evaluate older models
            if num_classes != 120 and num_classes != 125:
                self.mapping = make_class_mapping(self.video_list)
            else:
                self.mapping = None
            self.usable_objectives = None #this is what defines which workflow we are in for __getitem__
        else:
            self.usable_objectives = list()
            self.mappings = list()
            for i, (objective, objective_name) in enumerate(zip(num_classes, FromVideoDatasetLoader.OBJECTIVE_NAMES)):
                self.usable_objectives.append(objective > 0)
                if objective != VideoAndPointDatasetLoader.EPIC_MAX_CLASSES[i] and self.usable_objectives[-1]:
                    self.mappings.append(make_class_mapping_generic(self.video_list, objective_name))
                else:
                    self.mappings.append(None)
            assert any(obj is True for obj in self.usable_objectives)
        self.point_list_prefix = point_list_prefix
        self.image_tmpl = img_tmpl
        self.transform = batch_transform
        self.validation = validation
        self.norm_val = np.array(norm_val)
        self.vis_data = vis_data

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        frame_count = self.video_list[index].num_frames
        label_verb = self.video_list[index].label_verb
        label_noun = self.video_list[index].label_noun
        start_frame = self.video_list[index].start_frame
        start_frame = start_frame if start_frame != -1 else 0
        sampled_idxs = self.sampler.sampling(range_max=frame_count, v_id=index, start_frame=start_frame)
        # sampled_idxs = list(range(start_frame, start_frame + frame_count + 1))

        sampled_frames = load_images(self.video_list[index].data_path, sampled_idxs, self.image_tmpl)

        clip_input = np.concatenate(sampled_frames, axis=2)

        a, b, c, pid, vid_id = self.video_list[index].data_path.split("\\")
        track_path = os.path.join(self.point_list_prefix, pid, vid_id,
                                  "{}_{}_{}.pkl".format(start_frame, label_verb, label_noun))
        hand_tracks = load_pickle(track_path)
        # hand_tracks = load_pickle(self.samples_list[index].data_path)

        left_track = np.array(hand_tracks['left'], dtype=np.float32)
        right_track = np.array(hand_tracks['right'], dtype=np.float32)
        assert (self.video_list[index].num_frames + 1 == len(left_track)) # add + 1 because in the epic annotations the last frame is inclusive

        idxs = (np.array(sampled_idxs) - start_frame).astype(np.int)
        left_track = left_track[idxs]  # keep the points for the sampled frames
        right_track = right_track[idxs]
        left_track = left_track[::2]  # keep 1 coordinate pair for every two frames because we supervise 8 outputs from the temporal dim of mfnet and not 16 as the inputs
        right_track = right_track[::2]

        norm_val = self.norm_val
        if self.transform is not None:
            or_h, or_w, _ = clip_input.shape
            # for i in range(10):
            clip_input = self.transform(clip_input)
            is_flipped = False
            if 'RandomScale' in self.transform.transforms[0].__repr__(): # means we are in training so get the transformations
                sc_w, sc_h = self.transform.transforms[0].get_new_size()
                tl_y, tl_x = self.transform.transforms[1].get_tl()
                if 'RandomHorizontalFlip' in self.transform.transforms[2].__repr__():
                    is_flipped = self.transform.transforms[2].is_flipped()
            elif 'Resize' in self.transform.transforms[0].__repr__():  # means we are in testing
                sc_h, sc_w, _ = self.transform.transforms[0].get_new_shape()
                tl_y, tl_x = self.transform.transforms[1].get_tl()
            else:
                sc_w = or_w
                sc_h = or_h
                tl_x = 0
                tl_y = 0

            # apply transforms to tracks
            scale_x = sc_w/or_w
            scale_y = sc_h/or_h
            left_track *= [scale_x, scale_y]
            left_track -= [tl_x, tl_y]
            right_track *= [scale_x, scale_y]
            right_track -= [tl_x, tl_y]

            _,_, max_h, max_w = clip_input.shape
            norm_val = [max_w, max_h, max_w, max_h]
            if is_flipped:
                left_track[:, 0] = max_w - left_track[:, 0]
                right_track[:, 0] = max_w - right_track[:, 0]

            if self.vis_data:
                def vis_with_circle(img, left_point, right_point, winname):
                    k = cv2.circle(img.copy(), (int(left_point[0]), int(left_point[1])), 10, (255,0,0), 4)
                    k = cv2.circle(k, (int(right_point[0]), int(right_point[1])), 10, (0,0,255), 4)
                    cv2.imshow(winname, k)

                orig_left = np.array(hand_tracks['left'], dtype=np.float32)
                orig_left = orig_left[idxs]
                orig_right = np.array(hand_tracks['right'], dtype=np.float32)
                orig_right = orig_right[idxs]

                vis_with_circle(sampled_frames[-1], orig_left[-1], orig_right[-1], 'no augmentation')
                vis_with_circle(clip_input[:,-1,:,:].numpy().transpose(1,2,0), left_track[-1], right_track[-1], 'transformed')
                vis_with_circle(clip_input[:,-1,:,:].numpy().transpose(1,2,0), orig_left[-1], orig_right[-1], 'transf_img_not_coords')
                cv2.waitKey(0)

        # for the DSNT layer normalize to [-1, 1] for x and to [-1, 2] for y, which can get values greater than +1 when the hand is originally not detected
        left_track = (left_track * 2 + 1) / norm_val[:2] - 1
        right_track = (right_track * 2 + 1) / norm_val[2:] - 1
        # print("transformed:", left_track, "\n",right_track)
        # print("original:", (2*orig_left[::2]+1)/self.norm_val[:2]-1, "\n", (2*orig_right[::2]+1)/self.norm_val[2:]-1)

        points = np.concatenate((left_track[:, np.newaxis, :], right_track[:, np.newaxis, :]), axis=1).astype(np.float32)

        if self.usable_objectives is None: # old workflow only for verbs and hands
            if self.mapping:
                class_id = self.mapping[label_verb]
            else:
                class_id = label_verb

            if not self.validation:
                return clip_input, class_id, points
            else:
                return clip_input, class_id, points, self.video_list[index].uid #self.video_list[index].data_path.split("\\")[-1]
        else: # new multitask workflow
            # get the labels for the tasks
            labels = list()
            if self.usable_objectives[0]:
                pass # for now
                # action_id = self.video_list[index].label_action
                # if self.mappings[0]:
                #     action_id = self.mappings[0][action_id]
                # labels.append(action_id)
            if self.usable_objectives[1]:
                verb_id = self.video_list[index].label_verb
                if self.mappings[1]:
                    verb_id = self.mappings[1][verb_id]
                labels.append(verb_id)
            if self.usable_objectives[2]:
                noun_id = self.video_list[index].label_noun
                if self.mappings[2]:
                    noun_id = self.mappings[2][noun_id]
                labels.append(noun_id)

            labels = np.array(labels, dtype=np.float32)
            labels = np.concatenate((labels, points.flatten()))
            if not self.validation:
                return clip_input, labels
            else:
                return clip_input, labels, self.video_list[index].uid


class PointVectorSummedDatasetLoader(torchDataset):
    def __init__(self, list_file, max_seq_length=None, num_classes=120,
                 dual=False, validation=False):
        self.samples_list = parse_samples_list(list_file, DataLine)
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

        feat_size = 456 + 256
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
                vec[i:, 456 + yl] += 1
            if yr < 256:
                vec[i:, feat_addon + xr] += 1
                vec[i:, feat_addon + 456 + yr] += 1

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


class PointImageDatasetLoader(torchDataset):
    sum_seq_size = 0

    def __init__(self, list_file, batch_transform=None, norm_val=None,
                 validation=False):
        self.samples_list = parse_samples_list(list_file, DataLine)
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
            intensities = np.linspace(1., 0., i + 1)[::-1]
            point_imgs[left_track[:i + 1, 1], left_track[:i + 1, 0], i] = intensities
            point_imgs[right_track[:i + 1, 1], right_track[:i + 1, 0], i] = intensities

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
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_select2_56_nd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_brd\epic_rgb_train_1.txt"
    #video_list_file = r"D:\Code\hand_track_classification\splits\epic_rgb_select2_56_nd_brd\epic_rgb_train_1.txt"
    # video_list_file = r"D:\Code\hand_track_classification\vis_utils\21247.txt"
    #point_list_prefix = 'hand_detection_tracks_lr001'
    # video_list_file = r"D:\Code\hand_track_classification\splits\gtea_rgb\fake_split2.txt"
    video_list_file = r"splits\gtea_rgb_frames\test_split1.txt"

    import torchvision.transforms as transforms
    from utils.dataset_loader_utils import RandomScale, RandomCrop, RandomHorizontalFlip, RandomHLS, ToTensorVid, \
        Normalize, \
        Resize, CenterCrop
    mean_3d = [124 / 255, 117 / 255, 104 / 255]
    std_3d = [0.229, 0.224, 0.225]

    seed = 0
    train_transforms = transforms.Compose([
        RandomScale(make_square=True, aspect_ratio=[0.8, 1. / 0.8], slen=[224, 288], seed=seed),
        RandomCrop((224, 224), seed=seed), RandomHorizontalFlip(seed=seed), RandomHLS(vars=[15, 35, 25]),
        ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])
    test_transforms = transforms.Compose([Resize((256, 256), False), CenterCrop((224, 224)),
         ToTensorVid(), Normalize(mean=mean_3d, std=std_3d)])

    val_sampler = MiddleSampling(num=16)
    # val_sampler = RandomSampling(num=16, interval=2, speed=[1.0, 1.0], seed=seed)
    # loader = VideoAndPointDatasetLoader(val_sampler, video_list_file, point_list_prefix, num_classes=2,
    #                                     img_tmpl='frame_{:010d}.jpg', norm_val=[456., 256., 456., 256.],
    #                                     batch_transform=train_transforms, vis_data=True)
    # loader = FromVideoDatasetLoader(val_sampler, video_list_file, 'GTEA', [106, 0, 2], [106, 19, 53], batch_transform=train_transforms,
    #                                 extra_nouns=False, validation=True, vis_data=False)
    # loader = FromVideoDatasetLoaderGulp(val_sampler, video_list_file, 'GTEA', [106, 0, 2], [106, 19, 53],
    #                                     batch_transform=train_transforms, extra_nouns=False, validation=True,
    #                                     vis_data=True, use_hands=True, hand_list_prefix=r"D:\Code\epic-kitchens-processing\output\gtea_hand_trackslr005\clean")
    loader = VideoFromImagesDatasetLoader(val_sampler, video_list_file, 'GTEA', [106, 0, 2], [106, 19, 53],
                                          batch_transform=train_transforms, extra_nouns=False, validation=True,
                                          vis_data=True, use_hands=True,
                                          hand_list_prefix=r"gtea_hand_detection_tracks_lr005")

    for i in range(len(loader)):
        item = loader.__getitem__(i)
        print("\rItem {} ok".format(i))

#    from dataset_loader_utils import Resize, ResizePadFirst
#    
#    image = cv2.imread(r"..\hand_detection_track_images\P24\P24_08\90442_0_35.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
#
#    resize_only = Resize((224,224), False, cv2.INTER_CUBIC)
##    resize_pad = ResizeZeroPad(224, True, cv2.INTER_NEAREST)
#    cubic_pf_fun = ResizePadFirst(224, True, cv2.INTER_CUBIC)
#    linear_pf_fun = ResizePadFirst(224, True, cv2.INTER_LINEAR)
#    nearest_pf_fun = ResizePadFirst(224, True, cv2.INTER_NEAREST)
#    area_pf_fun = ResizePadFirst(224, True, cv2.INTER_AREA)
#    lanc_pf_fun = ResizePadFirst(224, True, cv2.INTER_LANCZOS4)
#    linext_pf_fun = ResizePadFirst(224, True, cv2.INTER_LINEAR_EXACT)
#
#    resize_nopad = resize_only(image)
##    resize_pad_first = resize_pad(image)
#    
#    cubic_pf = cubic_pf_fun(image)
##    cubic_pf = np.where(cubic_pf_fun(image) > 1, 255, 0).astype(np.float32)
##    nearest_pf = np.where(nearest_pf_fun(image) > 1, 255, 0).astype(np.uint8)
##    linear_pf = np.where(linear_pf_fun(image) > 1, 255 ,0).astype('uint8')
##    area_pf = np.where(area_pf_fun(image) > 1, 255 ,0).astype('uint8')
##    lanc_pf = np.where(lanc_pf_fun(image) > 1, 255, 0).astype('uint8')
#    linext_pf = linext_pf_fun(image)
#    
#    cv2.imshow('original', image)
#    cv2.imshow('original resize', resize_nopad)
##    cv2.imshow('padded resize', resize_pad_first)
#    cv2.imshow('cubic', cubic_pf)
##    cv2.imshow('nearest', nearest_pf)
##    cv2.imshow('area', area_pf)
##    cv2.imshow('lanc', lanc_pf)
#    cv2.imshow('linext', linext_pf)    
#    
#    cv2.waitKey(0)
#
