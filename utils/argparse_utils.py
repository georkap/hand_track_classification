# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 12:01:06 2018

argparse utils

@author: GEO
"""

import os, argparse

def make_base_parser(val):
    parser = argparse.ArgumentParser(description='Hand activity recognition')

    # Load the necessary paths    
    if not val:
        parser.add_argument('train_list', type=str)
        parser.add_argument('test_list', type=str)
        parser.add_argument('--base_output_dir', type=str, default=r'outputs/')
        parser.add_argument('--model_name', type=str, default=None, help='if left to None it will be automatically created from the args')
    else:
        parser.add_argument('ckpt_path', type=str)
        parser.add_argument('val_list', type=str)
        parser.add_argument('--annotations_path', type=str, default=None)
    parser.add_argument('--bpv_prefix', type=str, default='noun_bpv_oh', 
                        choices=['noun_bpv_oh', 'tracked_noun_bpv_oh_mh0', 'cln_noun_tracks_mh0', 'hand_detection_tracks',
                                 'hand_detection_tracks_lr', 'hand_detection_tracks_lr005', 'hand_detection_tracks_lr001'])
    parser.add_argument('--append_to_model_name', type=str, default="")
    
    return parser
    
def parse_args_dataset(parser, net_type):
    # Dataset parameters
    parser.add_argument('--verb_classes', type=int, default=120)
    parser.add_argument('--noun_classes', type=int, default=322)
    parser.add_argument('--batch_size', type=int, default=1)
    if net_type in ['resnet', 'mfnet']:
        parser.add_argument('--clip_gradient', action='store_true')
    if net_type == 'resnet':
        parser.add_argument('--no_resize', default=False, action='store_true', help='keep the original 256x456 image size for the network input')
        parser.add_argument('--inter', type=str, default='cubic',
                            choices=['linear', 'cubic', 'nn', 'area', 'lanc', 'linext'])
        parser.add_argument('--bin_img', default=False, action='store_true')
        parser.add_argument('--pad', default=False, action='store_true')
    if net_type == 'mfnet':
        parser.add_argument('--clip_length', type=int, default=16, help="define the length of each input sample.")
        parser.add_argument('--frame_interval', type=int, default=2, help="define the sampling interval between frames.")
        #parser.add_argument('--img_tmpl', type=str)
    if net_type in ['lstm', 'lstm_polar', 'lstm_diffs']:
        parser.add_argument('--lstm_feature', default='coords',
                            choices=['coords', 'coords_bpv', 'coords_objects', 'coords_dual', 'coords_polar','coords_diffs','vec_sum', 'vec_sum_dual'],
                            help="lstm_input changes based on the choice."\
                            + "coords: 4,"\
                            + "coords with only_left/only_right: 2,"\
                            + "coords_bpv: 356," \
                            + "coords_objects: 708," \
                            + "coords_dual: 2," \
                            + "coords_polar: 8 for coords+dists+angles, 2 for angles," \
                            + "coords_diffs: 8 for coords+diffs," \
                            + "vec_sum/vec_sum_dual: 712 (i.e. 256+456)")
        parser.add_argument('--no_norm_input', default=False, action='store_true', help='will not normalize input to 0-1')
    if net_type == 'lstm':
        parser.add_argument('--lstm_clamped', default=False, action='store_true', 
                            help='will remove the non existing hand points in a sequence and result in each hand having a starting sequence of different length. Sampling from these sequences is possible afterwards. Works only for dual lstm and coords feature.')
    
    return parser
    
def parse_args_network(parser, net_type):
    # Network configuration
    if net_type in ['resnet', 'mfnet']:
        parser.add_argument('--pretrained', default=False, action='store_true')
    if net_type == 'resnet':
        parser.add_argument('--resnet_version', type=str, default='18', 
                            choices=['18','34','50','101','152'], 
                            help="One of 18, 34, 50, 101, 152")
        parser.add_argument('--channels', default='RGB', choices=['RGB', 'G'], help="optional to train on one input channel with binary inputs.")
        parser.add_argument('--feature_extraction', default=False, action='store_true')
    if net_type == 'mfnet':
        parser.add_argument('--pretrained_model_path', type=str, default=r"models\MFNet3D_Kinetics-400_72.8.pth")
    if net_type in ['lstm', 'lstm_polar', 'lstm_diffs']:
        parser.add_argument('--lstm_bidir', default=False, action='store_true')
        parser.add_argument('--lstm_input', type=int, default=4)
        parser.add_argument('--lstm_hidden', type=int, default=8)
        parser.add_argument('--lstm_layers', type=int, default=2)
        parser.add_argument('--lstm_seq_size', type=int, default=0, help="If not 0, it will perform a uniform sampling over the sequence to get to the desired number.")
    if net_type == 'lstm':
        parser.add_argument('--lstm_dual', default=False, action='store_true')
        parser.add_argument('--lstm_attn', default=False, action='store_true')
        parser.add_argument('--only_left', default=False, action='store_true')
        parser.add_argument('--only_right', default=False, action='store_true')
        
    parser.add_argument('--double_output', default=False, action='store_true')
    
    return parser
    
def parse_args_training(parser):
    # Training parameters
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--mixup_a', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_mult_base', type=float, default=1.0)
    parser.add_argument('--lr_mult_new', type=float, default=1.0)
    parser.add_argument('--lr_type', type=str, default='step',
                        choices=['step', 'multistep', 'clr', 'groupmultistep'])
    parser.add_argument('--lr_steps', nargs='+', type=str, default=[7],
                        help="The value of lr_steps depends on lr_type. If lr_type is:"\
                            +"'step' then lr_steps is a list of size 2 that contains the number of epochs needed to reduce the lr at lr_steps[0] and the gamma to reduce by, at lr_steps[1]."\
                            +"'multistep' then lr_steps is a list of size n+1 for n number of learning rate decreases and the gamma to reduce by at lr_steps[-1]."\
                            +"'clr' then lr_steps is a list of size 6: [base_lr, max_lr, num_epochs_up, num_epochs_down, mode, gamma]. In the clr case, argument 'lr' is ignored."\
                            +"'groupmultistep' then the arguments are used like 'multistep' but internally different learning rate is applied to different parameter groups.")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005) # decay for mfnet is 0.0001
    parser.add_argument('--max_epochs', type=int, default=20)
    
    return parser

def parse_args_eval(parser):
    # Parameters for evaluation during training
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--eval_on_train', default=False, action='store_true')
    # Parameters for evaluation during testing
    # mfnet
    parser.add_argument('--mfnet_eval', type=int, default=1)
    parser.add_argument('--eval_sampler', type=str, default='random', choices=['middle', 'random'])
    parser.add_argument('--eval_crop', type=str, default='random', choices=['center', 'random'])
    # lstm
    parser.add_argument('--save_attentions', default=False, action='store_true')

    return parser

def parse_args_program(parser):
    # Program parameters
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_all_weights', default=False, action='store_true')
    parser.add_argument('--logging', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_from', type=str, default="", help="specify where to resume from otherwise resume from last checkpoint")
    
    return parser
    
def parse_args(net_type, val=False):
    parser = make_base_parser(val)
    parser = parse_args_dataset(parser, net_type)
    parser = parse_args_network(parser, net_type)
    parser = parse_args_training(parser)
    parser = parse_args_eval(parser)
    parser = parse_args_program(parser)
    
    args = parser.parse_args()
    if not val:
        if args.model_name is None:
            model_name = make_model_name(args, net_type)        
            return args, model_name
        return args, args.model_name
    return args


def make_model_name(args, net_type):
    if net_type == 'resnet':
        model_name = "res{}_{}_{}_{}_{}".format(args.resnet_version, args.channels, args.batch_size,
                                                str(args.dropout).split('.')[0]+str(args.dropout).split('.')[1],
                                                args.max_epochs)
        if args.pretrained:
            model_name = model_name + "_pt"
        if args.feature_extraction:
            model_name = model_name + "_ft"
        
        model_name = model_name + "_{}".format(args.inter)
        if args.bin_img:
            model_name = model_name + "_bin"
        else:
            model_name = model_name + "_smooth"
        if args.pad:
            model_name = model_name + "_pad"
        else:
            model_name = model_name + "_nopad"
    if net_type == 'mfnet':
        model_name = "{}_{}_{}_{}_cl{}".format(net_type, args.batch_size,
                                               str(args.dropout).split('.')[0]+str(args.dropout).split('.')[1],
                                               args.max_epochs, args.clip_length)
        if args.pretrained:
            model_name = model_name + "_pt"
    if net_type in ['lstm', 'lstm_polar', 'lstm_diffs']: # not connecting the two cases until I change the result parser to support new and old cases together
        ntype = net_type + "b" if args.lstm_bidir else net_type
            
        model_name = "{}_{}_{}_{}".format(ntype, args.batch_size,
                      str(args.dropout).split('.')[0]+str(args.dropout).split('.')[1],
                      args.max_epochs)
        model_name = model_name + "_{}_{}_{}".format(args.lstm_input,
                                    args.lstm_hidden, args.lstm_layers)
        model_name = model_name + "_seq{}".format(args.lstm_seq_size if args.lstm_seq_size != 0 else "full")
        model_name = model_name + "_{}".format(args.lstm_feature)
    
    if net_type == 'lstm': # each type a feature starts being supported from all cases move up
        if args.lstm_dual:
            model_name = model_name + "_dual"
        if args.lstm_attn:
            model_name = model_name + "_attn"
        if args.lstm_clamped:
            model_name = model_name + "_clamped"
        if args.no_norm_input:
            model_name = model_name + "_no_norm"
        if args.only_left:
            model_name = model_name + "_onlyleft"
        if args.only_right:
            model_name = model_name + "_onlyright"
        
# keep this for legacy purpose when I will need to change the parser
#    if net_type == 'lstm':
#        model_name = "lstm"
#        model_name = model_name + "_{}_{}_{}".format(args.batch_size, 
#                                    args.dropout, args.max_epochs)
#        if args.lstm_dual:
#            model_name = model_name + "_dual"
#        if args.lstm_attn:
#            model_name = model_name + "_attn"
#        model_name = model_name + "_{}_{}_{}".format(args.lstm_input,
#                                    args.lstm_hidden, args.lstm_layers)
#        model_name = model_name + "_seq{}".format(args.lstm_seq_size if args.lstm_seq_size != 0 else "full")
#        model_name = model_name + "_{}".format("coords" if args.lstm_feature.startswith('coord') else "vec")
#        if args.lstm_clamped:
#            model_name = model_name + "_clamped"
#        if args.no_norm_input:
#            model_name = model_name + "_no_norm"
#        if args.only_left:
#            model_name = model_name + "_onlyleft"
#        if args.only_right:
#            model_name = model_name + "_onlyright"
    
    model_name = model_name + "_{}".format(args.lr_type)
    if args.lr_type == "clr":
        clr_type = "tri" if args.lr_steps[4] == "triangular" else "tri2" if args.lr_steps[4] == "triangular2" else "exp"
        model_name = model_name + "_{}".format(clr_type)
        model_name = model_name + str(args.lr_steps[0]).split('.')[0] + str(args.lr_steps[0]).split('.')[1] + '-' + str(args.lr_steps[1]).split('.')[0] + str(args.lr_steps[1]).split('.')[1]
    model_name = model_name + "_vsel{}".format(args.verb_classes)
    if args.double_output:
        model_name = model_name + "_nsel{}".format(args.noun_classes)
    if args.mixup_a != 1.:
        model_name = model_name + "_mixup" 

    model_name = model_name + args.append_to_model_name
    
    return model_name

def make_log_file_name(output_dir, args):
    # creates the file name for the evaluation log file or None if no logging is required
    if args.logging:
        log_file = os.path.join(output_dir, "results-accuracy-validation")
        if args.double_output:
            if 'verb' in args.ckpt_path:
                log_file = os.path.join(output_dir, "results-accuracy-validation-verb")
            if 'noun' in args.ckpt_path:
                log_file = os.path.join(output_dir, "results-accuracy-validation-noun")
        log_file += args.append_to_model_name
        log_file += ".txt"
    else:
        log_file = None
    return log_file

def parse_args_train_log_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log_path", type=str, help="The file to be analyzed", required=True)
    parser.add_argument("--no_lr_graph", default=False, action='store_true')
    parser.add_argument("--coord_loss", default=False, action='store_true')
    return parser.parse_args()

def parse_args_train_log_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log_dir", type=str, help="Dir containing the logs to be analyzed", required=True)
    parser.add_argument("--walk", default=False, action='store_true')
    return parser.parse_args()

def parse_args_val():
    ''' Only for legacy experiments'''
    parser = argparse.ArgumentParser(description='Hand activity recognition - validation')
    
    # Load the necessary paths
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('val_list', type=str)
    
    # Network configuration
    parser.add_argument('--resnet_version', type=str, default=None, 
                        choices=['18','34','50','101','152'], 
                        help="One of 18, 34, 50, 101, 152")
    parser.add_argument('--channels', default=None, choices=['RGB', 'G'], 
                        help="optional to train on one input channel with binary inputs.")
    parser.add_argument('--lstm_input', type=int, default=4)
    parser.add_argument('--lstm_hidden', type=int, default=8)
    parser.add_argument('--lstm_layers', type=int, default=2)

    # Dataset parameters
    parser.add_argument('--verb_classes', type=int, default=120)
    parser.add_argument('--no_resize', default=False, action='store_true')
    parser.add_argument('--inter', type=str, default='cubic',
                        choices=['linear', 'cubic', 'nn', 'area', 'lanc', 'linext'])
    parser.add_argument('--bin_img', default=False, action='store_true')
    parser.add_argument('--pad', default=False, action='store_true')    
    parser.add_argument('--no_norm_input', default=False, action='store_true', help='lstm')
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Program parameters
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--logging', default=False, action='store_true')

    args = parser.parse_args()
    return args