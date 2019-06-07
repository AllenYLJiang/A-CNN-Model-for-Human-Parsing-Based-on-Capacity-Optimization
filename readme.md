Three steps:
1. Concatenate depth (Pascal_Depth provided by us) with rgb images (JPEGImages provided by us) to feed into network (dump_RGB_depth_model.prototxt) for test. Pretrained model is provided in 

2. Train original model which is based on only rgb images. 

3. Fuse the predictions from 1 and 2 with fuse_depth_accuracy_with_1class_segmentation.py
