Three steps:
1. Concatenate depth (Pascal_Depth provided by us) with rgb images (JPEGImages provided by us) to feed into network (test_RGB_depth_model.prototxt dump_RGB_depth_model.prototxt) for test. Pretrained model is provided in 
https://drive.google.com/file/d/1vAOfY2nLf7eppZqE8Dn6nTy5Bdpu_hn6/view?usp=sharing
Depth estimations: https://drive.google.com/file/d/1O06JgxwEM-sTMyUs-qjKZYvhIQ_8GW_i/view?usp=sharing
JPEGImages: https://drive.google.com/file/d/1_-z5UOMDxRAHdVz690W9lahADnN0GhMS/view?usp=sharing
ground truth: https://drive.google.com/file/d/1TYCyJ08pP5MBiBVGEb2-xVTa6vBcPY2j/view?usp=sharing

2. Train original model which is based on only rgb images. 
test_RGB_model.prototxt
dump_RGB_model.prototxt
Pretrained model is provided in 
https://drive.google.com/file/d/1Bzujy9iXPdw8CUT3Ds462DSt1p0OCjcv/view?usp=sharing

3. Fuse the predictions from 1 and 2 with fuse_depth_accuracy_with_1class_segmentation.py
Method for fusion: 
We've trained a segmentation models with one foreground class:
https://drive.google.com/file/d/1Dklqh6DU92C2_9j3SdD9mYSuLHnY5FgB/view?usp=sharing
segmentation_1_foreground_class.prototxt 
We need to check which one of the two predictions on each image (with or without depth) is closer to the 1-class prediction. Here the metric is miou. Then for each image we adopt the prediction from the model (with or without depth) whose prediction is more similar to that of 1 class segmentation model.
The predictions from the 1-class segmentation model:
https://drive.google.com/file/d/1bEehmh5dQUD18et_FeGEJESVbbC4S7S4/view?usp=sharing

To further improve performance, refer to 

<img width="600" height="208" src="https://github.com/AllenYLJiang/A-CNN-Model-for-Human-Parsing-Based-on-Capacity-Optimization/blob/master/Stage1.png"/> 
