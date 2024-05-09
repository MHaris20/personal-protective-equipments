import sys
import cv2
import time
import argparse
import numpy as np
from numpy import random
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
from utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
    non_max_suppression, apply_classifier, scale_boxes, xyxy2xywh, strip_optimizer, set_logging, increment_path

from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


def detect(model,
           device,
           imgsz,
           image,
           half,
           conf,
           stride):
    # model = model.to(device) 
    # stride, names, pt = model.stride, model.names, model.pt  # model stride
    # names = model.module.names if hasattr(
    #     model, 'module') else model.names  # get class names
    # if half:
    #     model.half()  # to FP16
        
    # print("-------- GPU:",next(model.parameters()).device)

    
    imgsz = check_img_size(imgsz, s=stride)
   
    im0 = image.copy()
   
    img = letterbox(image, imgsz, stride=stride, auto=True)[0]
    names = model.module.names if hasattr(model, 'module') else model.names
    ascii = is_ascii(names)
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    dt, seen = [0.0, 0.0, 0.0], 0
    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1
    pred = model(img, augment=False, visualize=False)[0]
    t3 = time_sync()
    dt[1] += t3 - t2

    pred = non_max_suppression(pred, conf, 0.10, None, False, max_det=1000)
    # print(pred)
    dt[2] += time_sync() - t3
    for i, det in enumerate(pred):
        seen += 1
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # norma
        display_str_list = []
        display_str_dict = {}
        annotator = Annotator(im0, line_width=4, pil=not ascii)
        if len(det):
            #   # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(
                img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if False else (
                    names[c] if True else f'{names[c]} {conf:.2f}')  # Label
                annotator.box_label(xyxy, label, color=colors(c, True))
                label = f'{names[int(cls)]} {conf:.2f}'
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
                display_str_dict = {
                    'name': names[int(cls)],
                    'score': f'{conf:.2f}',
                    'ymin': y1,
                    'xmin': x1,
                    'ymax': y2,
                    'xmax': x2,
                    'image': image[y1:y2, x1:x2]}
                display_str_list.append(display_str_dict)

        return im0, display_str_list
