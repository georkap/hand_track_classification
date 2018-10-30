REM BATCH RUN TRAIN CONFIGURATIONS

REM variations for linear : no pretrained + nopad, pretrained + nopad, pretrained + pad, pretrained + binary + pad 

python main_resnet.py res34_RGB_256_05_20_linear_smooth_nopad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter linear --resnet_version 34 --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_linear_smooth_nopad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter linear --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_linear_smooth_pad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter linear --pad --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_linear_bin_pad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter linear --bin_img --pad --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging

REM variations for cubic: no pretrained + nopad, pretrained + nopad, pretrained + pad, pretrained + binary + pad 

python main_resnet.py res34_RGB_256_05_20_cubic_smooth_nopad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter cubic --resnet_version 34 --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_cubic_smooth_nopad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter cubic --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_cubic_smooth_pad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter cubic --pad --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_cubic_bin_pad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter cubic --bin_img --pad --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging

REM variations for nearest neighbor: no pretrained + nopad, pretrained + nopad, pretrained + pad, pretrained + binary + pad

python main_resnet.py res34_RGB_256_05_20_nn_smooth_nopad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter nn --resnet_version 34 --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_nn_smooth_nopad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter nn --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_nn_smooth_pad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter nn --pad --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_nn_bin_pad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter nn --bin_img --pad --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging

REM variations for linear bit exact (to compare against plain linear): no pretrained + nopad, pretrained + nopad, pretrained + pad, pretrained + binary + pad 

python main_resnet.py res34_RGB_256_05_20_linext_smooth_nopad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter linext --resnet_version 34 --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_linext_smooth_nopad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter linext --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_linext_smooth_pad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter linext --pad --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging
python main_resnet.py res34_RGB_256_05_20_pt_linext_bin_pad splits\hand_track_images\hand_track_train_1.txt splits\hand_track_images\hand_track_val_1.txt --inter linext --bin_img --pad --resnet_version 34 --pretrained --batch_size 256 --num_workers 16 --eval_on_train --logging
