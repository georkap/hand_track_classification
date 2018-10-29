# hand_track_classification

This repository is used for classification of hand tracking images from egocentric videos of the Epic Kitchens [PDF](https://arxiv.org/pdf/1804.02748.pdf) dataset. 
Using pytorch 0.4 currently

## Steps taken
I used tracking of hands by detection with a Yolov3 hand detector [Git](https://github.com/AlexeyAB/darknet) and the Sort tracker [PDF](https://arxiv.org/pdf/1602.00763.pdf) to fill in the gaps in the detections, 
in order to produce continuous hand tracks.
Then I turn the hand tracks into images which I classify using a resnet (18, 50, 101) backbone into the 120 verb classes of Epic Kitchens.

## Data Loading
It is practical to create symlinks "mklink \D target origin" into the base folder and have the split\*.txt files point to the relative path to the base folder. 

TODO list:
- [ ] Add code to produce 3d representation of tracks
- [ ] Integrate MFNet for 3d convolutions
