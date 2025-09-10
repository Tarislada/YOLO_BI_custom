import ultralytics
from ultralytics import YOLO
# import wandb
# from wandb.integration.ultralytics import add_wandb_callback
import torch
import numpy as np

# model = YOLO('YOLO_custom/Models/KH_bot_ir/KH_bot_sv3b.pt')
# model = YOLO('/home/tarislada/YOLOprojects/YOLO_custom/Models/KH_binocular/KH_NoseNtail_sv3.pt')
# model = YOLO('yolo12s-pose.yaml')
# model = YOLO('Models/Cricket_detection/Cricket_v11s_04.pt')
# model = YOLO('/home/tarislada/YOLOprojects/YOLO_custom/Models/YW/Tremor_12sv02/weights/best.pt')
# mode = YOLO('YOLO_custom/Models/best_setup3.pt') 
# model=YOLO('yolo12-pose.yaml')
# model = YOLO('yolo12s-seg.pt') # Load a pre-trained YOLOv12 model for segmentation
model = YOLO('/home/tarislada/YOLOprojects/YOLO_custom/Models/Nat_segment.pt')
# add_wandb_callback(model, enable_model_checkpointing=True)

# data = '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/takehome/KH.yaml' 
# data = '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/takehome/KH_OD.yaml'
# data = '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/takehome/YW.yaml'
data = '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Nat/Nat_seg.yaml'
# model.train(data=data, epochs=500, patience=250,imgsz=900,dfl=2.5,pretrained=True,scale=0.25) # KH-cricket traincode train model with data. 
# model.train(data=data, imgsz = (480, 720), epochs=500, patience=500, pose=16, pretrained=True, hsv_v=0.6, shear=10, perspective=0.0001, batch=0.8) # YW training code
model.train(data=data, epochs = 500, imgsz = (1080,1080), batch=0.8, hsv_h=0.3, hsv_s=0.8, hsv_v=0.8, translate=0.5, shear=5, perspective=0.00025, copy_paste=0.2, copy_paste_mode='mixup')
# model.train(data=data, epochs=2000, patience=2000)
# model.train(data=data, imgsz=1180, epochs=200, patience=150, pose = 14) # KH-pose traincode
# model.train(data=data, imgsz=1120, epochs=300, patience=150, pose = 14) # KH-pose traincode
# model.train(data=data, imgsz=(640, 640), epochs=300, patience=150, hsv_s=) # KH-pose traincode
model.val(data = data) # validate the model with the data. use validation dataset.

# wandb.finish()

# model = YOLO('/home/tarislada/YOLOprojects/YOLO_custom/Models/MFDS_OD_v1.pt')
# # results = model.predict('/home/tarislada/YOLOprojects/YOLO_custom/testvid.mp4',save_txt=True,stream=True,)
# results = model('/home/tarislada/YOLOprojects/YOLO_custom/230824_Thiopropamine_10mg-4.mp4',stream=True,verbose=False)
