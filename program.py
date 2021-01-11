# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:47:44 2020
https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/
https://www.youtube.com/watch?v=XNRzZkZ-Byg
https://colab.research.google.com/drive/1e4zvS6LyhOAayEDh3bz8MXFTJcVFSvZX?usp=sharing
@author: alebj
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            rtmp://192.168.1.105/live/test  # rtmp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
"""
import os
os.chdir("C:\\Users\\alebj\\Documents\\Python Scripts\\YoloV5-Object-Detection")

import torch
from IPython.display import Image  # for displaying images
from utils.google_utils import gdrive_download  # for downloading models/datasets

print('torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

#python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/Image(filename='runs/detect/exp/zidane.jpg', width=600)
#python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/zidane.jpg
#python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source 0 #Web cam
#python detect.py --source video.mp4

os.system("python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/zidane.jpg")
