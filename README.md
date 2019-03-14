# hand_track_classification

This repository is used for classification of hand tracking images from egocentric videos of the Epic Kitchens [PDF](https://arxiv.org/pdf/1804.02748.pdf) [Project page](https://epic-kitchens.github.io/2018) dataset. 
Using pytorch 0.4 currently

## Steps taken
We used tracking of hands by detection with a Yolov3 hand detector [Git](https://github.com/AlexeyAB/darknet) and the Sort tracker [PDF](https://arxiv.org/pdf/1602.00763.pdf) to fill in the gaps in the detections, 
in order to produce continuous hand tracks.
Then we turn the hand tracks into images which we classify using a resnet (18, 50, 101) backbone into the 120 verb classes of Epic Kitchens.
These pipelines are covered by ```main_cnn``` and its variations

The points per frame of each track can also be used as input to an lstm and be classified into the verb classes of Epic Kitchens
These pipelines are covered by ```main_lstm``` and its variations 

For practical reasons for my project I added support for the MFNET network [Git](https://github.com/cypw/PyTorch-MFNet) [PDF](https://arxiv.org/abs/1807.11195) to train directly on the RGB images of Epic Kitchens
without direct hand support yet. Pipelines are covered by ```main_mfnet``` and its variations

## Usage

### Data Loading
It is practical to create symlinks "mklink /D target origin" into the base folder and have the split\\*.txt files point to the relative path to the base folder. 

### Create split files
.txt files that register the locations of the actual files that are used for training along with other pieces of information e.g. class, number of frames etc.

### Arguments
For a comprehensive list of arguments see ```utils\argparse_utils.py ```

Examples:
CNN training
```usage: main_cnn.py [-h] [--base_output_dir BASE_OUTPUT_DIR]
                   [--model_name MODEL_NAME] [--verb_classes VERB_CLASSES]
                   [--noun_classes NOUN_CLASSES] [--batch_size BATCH_SIZE]
                   [--clip_gradient] [--no_resize]
                   [--inter {linear,cubic,nn,area,lanc,linext}] [--bin_img]
                   [--pad] [--pretrained]
                   [--resnet_version {18,34,50,101,152}] [--channels {RGB,G}]
                   [--feature_extraction] [--double_output]
                   [--dropout DROPOUT] [--mixup_a MIXUP_A] [--lr LR]
                   [--lr_type {step,multistep,clr,groupmultistep}]
                   [--lr_steps LR_STEPS [LR_STEPS ...]] [--momentum MOMENTUM]
                   [--decay DECAY] [--max_epochs MAX_EPOCHS]
                   [--gpus GPUS [GPUS ...]] [--num_workers NUM_WORKERS]
                   [--eval_freq EVAL_FREQ] [--mfnet_eval MFNET_EVAL]
                   [--eval_on_train] [--save_all_weights] [--save_attentions]
                   [--logging] [--resume]
                   train_list test_list
```
LSTM training
```
usage: main_lstm.py [-h] [--base_output_dir BASE_OUTPUT_DIR]
                    [--model_name MODEL_NAME] [--verb_classes VERB_CLASSES]
                    [--noun_classes NOUN_CLASSES] [--batch_size BATCH_SIZE]
                    [--lstm_feature {coords,coords_dual,coords_polar,coords_diffs,vec_sum,vec_sum_dual}]
                    [--no_norm_input] [--lstm_clamped]
                    [--lstm_input LSTM_INPUT] [--lstm_hidden LSTM_HIDDEN]
                    [--lstm_layers LSTM_LAYERS]
                    [--lstm_seq_size LSTM_SEQ_SIZE] [--lstm_dual]
                    [--lstm_attn] [--only_left] [--only_right]
                    [--double_output] [--dropout DROPOUT] [--mixup_a MIXUP_A]
                    [--lr LR] [--lr_type {step,multistep,clr,groupmultistep}]
                    [--lr_steps LR_STEPS [LR_STEPS ...]] [--momentum MOMENTUM]
                    [--decay DECAY] [--max_epochs MAX_EPOCHS]
                    [--gpus GPUS [GPUS ...]] [--num_workers NUM_WORKERS]
                    [--eval_freq EVAL_FREQ] [--mfnet_eval MFNET_EVAL]
                    [--eval_on_train] [--save_all_weights] [--save_attentions]
                    [--logging] [--resume]
                    train_list test_list
```
MFNet training
```
usage: main_mfnet.py [-h] [--base_output_dir BASE_OUTPUT_DIR]
                     [--model_name MODEL_NAME] [--verb_classes VERB_CLASSES]
                     [--noun_classes NOUN_CLASSES] [--batch_size BATCH_SIZE]
                     [--clip_gradient] [--clip_length CLIP_LENGTH]
                     [--frame_interval FRAME_INTERVAL] [--pretrained]
                     [--pretrained_model_path PRETRAINED_MODEL_PATH]
                     [--double_output] [--dropout DROPOUT] [--mixup_a MIXUP_A]
                     [--lr LR] [--lr_type {step,multistep,clr,groupmultistep}]
                     [--lr_steps LR_STEPS [LR_STEPS ...]]
                     [--momentum MOMENTUM] [--decay DECAY]
                     [--max_epochs MAX_EPOCHS] [--gpus GPUS [GPUS ...]]
                     [--num_workers NUM_WORKERS] [--eval_freq EVAL_FREQ]
                     [--mfnet_eval MFNET_EVAL] [--eval_on_train]
                     [--save_all_weights] [--save_attentions] [--logging]
                     [--resume]
                     train_list test_list
```

## Contact
Georgios Kapidis
georgios{dot}kapidis{at}noldus{dot}nl
g{dot}kapidis{at}uu{dot}nl
georgios{dot}kapidis{at}gmail{dot}nl