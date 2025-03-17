import ultralytics
from ultralytics import YOLO
# import wandb
# from wandb.integration.ultralytics import add_wandb_callback
import torch
import numpy as np

model = YOLO('YOLO_custom/Models/KH_bot_ir/KH_bot_sv3b.pt')
# model = YOLO('/home/tarislada/YOLOprojects/YOLO_custom/Models/KH_binocular/KH_NoseNtail_sv3.pt')
# model = YOLO('yolov10n.pt')
# model = YOLO('YOLO_custom/Models/Cricket_detection/KH_cricket_v02.pt')
# mode = YOLO('YOLO_custom/Models/best_setup3.pt') 
# model=YOLO('yolo12-pose.yaml')
# add_wandb_callback(model, enable_model_checkpointing=True)

data = '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/takehome/KH.yaml'
# data = '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/takehome/KH_OD.yaml'
# model.train(data=data, epochs=250, patience=250,imgsz=900,dfl=2.5,pretrained=True) # KH-cricket traincode train model with data. 
# model.train(data=data, epochs=2000, patience=2000)
# model.train(data=data, imgsz=1180, epochs=200, patience=150, pose = 14) # KH-pose traincode
model.train(data=data, imgsz=1120, epochs=300, patience=150, pose = 14) # KH-pose traincode
model.val(data = data) # validate the model with the data. use validation dataset.

# wandb.finish()

# model = YOLO('/home/tarislada/YOLOprojects/YOLO_custom/Models/MFDS_OD_v1.pt')
# # results = model.predict('/home/tarislada/YOLOprojects/YOLO_custom/testvid.mp4',save_txt=True,stream=True,)
# results = model('/home/tarislada/YOLOprojects/YOLO_custom/230824_Thiopropamine_10mg-4.mp4',stream=True,verbose=False)
