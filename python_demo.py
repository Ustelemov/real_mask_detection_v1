import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import copy
import os
import sys
import glob 
import argparse
import time

import urllib.request

import uuid

from time import strftime
from time import gmtime

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, strip_optimizer, set_logging
from utils.torch_utils import select_device, load_classifier, time_synchronized

from sort import Sort


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def detect(model, orgimg, device):
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    ret = []

    # Process detections
    # Add 30% of width\height padding for face
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
            
            for j in range(det.size()[0]):
                conf = det[j, 4].cpu().numpy()
                x_top_left = int(det[j,0])
                y_top_left = int(det[j,1])
                x_bottom_right = int(det[j,2])
                y_bottom_right = int(det[j,3])

                width = x_bottom_right - x_top_left
                height = y_bottom_right - y_top_left

                x_top_left = int(x_top_left - width*0.3)
                y_top_left = int(y_top_left - height*0.3)
                x_bottom_right = int(x_bottom_right + width*0.3)
                y_bottom_right = int(y_bottom_right + height*0.3)

                x_top_left = 0 if x_top_left < 0 else x_top_left
                y_top_left = 0 if y_top_left < 0 else y_top_left
                x_bottom_right = w0 if x_bottom_right > w0 else x_bottom_right
                y_bottom_right = h0 if y_bottom_right > h0 else y_bottom_right

                ret.append([x_top_left, y_top_left, x_bottom_right, y_bottom_right, conf])
    return ret

def mkdir(path):
    path.strip()
    path.rstrip('\\')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

class FaceData:
    def __init__(self, save_path, input_FPS):
        self.input_FPS = input_FPS
        self.faces = dict()
        self.save_path = save_path
        self.detections_count_threshold = 10
        self.detections_score_threshold = 0.7
        self.detections_max_score_threshold = 0.75

        self.saved_count=0

    def add(self, id, arr, score, frame_num):
        if id not in self.faces:
            self.faces[id] = []
        self.faces[id].append((arr,score, frame_num))
    
    def get_count(self,id):
        if id not in self.faces:
            return 0
        return len(self.faces[id])

    def remove(self, id):
        if id not in self.faces:
            return
        
        dataArr = self.faces[id]

        #remove element
        del self.faces[id]

        #if not enought detections
        if len(dataArr) < self.detections_count_threshold:
            return

        max_score = 0

        score_threshold_count = 0
        img = []
        frame_num = 0

        for data in dataArr[:]:
            if data[1] > max_score:
                img = data[0]
                max_score = data[1]
                frame_num = data[2]
            if data[1] > self.detections_score_threshold:
                score_threshold_count += 1
            
        #if max_score not enough
        if max_score < self.detections_max_score_threshold:
            return

        #if not enough detections with threshold score
        if score_threshold_count < self.detections_count_threshold:
            return

        mkdir(self.save_path)

        h,w,c = img.shape
        id = str(uuid.uuid1())
        frame_time = strftime("%H:%M:%S", gmtime(int(frame_num/self.input_FPS)))

        cv2.imwrite("{0}/{1}_{2}_{3}.jpg".format(self.save_path, id, w,h), img)

        #log for debug
        self.saved_count += 1
        print("---------saved id: %s (№ in dataset %s)---------"%(id, self.saved_count))
        print("max_score: ",max_score)
        print("all_count: ",len(dataArr))
        print("score_threshold_count: ",score_threshold_count)
        print("time: ",frame_time)
        for data in dataArr:
            print("score: %0.2f shape: %s"%(data[1], data[0].shape))
        print("---------------------------------------------------------------------------------")

def process_image(image, predicts, frame_num):
    h,w,c = image.shape
    face_size_threshold = 10

    result_image = image.copy()
    faces = []
    for predict in predicts:
        x1, y1, x2, y2, score = predict     

        if x2-x1 > face_size_threshold and y2-y1 > face_size_threshold:
            faces.append(predict)

    trackers, removed = tracker.update(np.array(faces))
    
    # visualize and add
    for d in trackers:
        score = d[5]
        d = d.astype(np.int32)

        cv2.rectangle(result_image, (d[0], d[1]), (d[2], d[3]), (0,0,0), 3)
        cv2.putText(result_image, 'ID%d (%0.2f) (%d)' % (d[4],score, face_data.get_count(d[4])+1), 
            (d[0] - 5, d[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)

        # add after check for bounds
        x1 = 0 if d[0] < 0 else d[0]
        y1 = 0 if d[1] < 0 else d[1]
        x2 = w if d[2] > w else d[2]
        y2 = h if d[3] > h else d[3]


        face = image[y1:y2, x1:x2]
        face_data.add(d[4],face, score,frame_num)

    # process removed
    for rm in removed:
        face_data.remove(rm)

    return result_image

if __name__ == '__main__':
    mkdir("output")

    # Model
    img_size = 320
    conf_thres = 0.2
    iou_thres = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('./weights/yolov5s-face.pt', device)

    #input video
    input_video = './input/input.mp4'
    cap = cv2.VideoCapture(input_video)
    length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    input_FPS = int(cap. get(cv2. CAP_PROP_FPS))
    print("Input video: %s FPS: %d"%(input_video, input_FPS))

    #output video
    output_video = './output/output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, input_FPS, (400,300))

    #tracker and data saver
    save_face_path = 'output/saved_faces'
    tracker = Sort(max_age=15, min_hits=0) 
    face_data = FaceData(save_path = save_face_path, input_FPS=input_FPS)

    #for print
    frame_count = 0
    count_time = time.time()
    all_time = time.time()
    count_per_frames = 1000

    while (cap.isOpened()):
        ret,frame = cap.read()
        if ret:

            #print
            frame_count = frame_count + 1
            if frame_count%count_per_frames == 0:
                print("---------------------------------------------------------------------------------")
                print("----------------------------", frame_count," of ", length, "-----------------------------")
                print("---------------------------------------------------------------------------------")
                print("FPS (for %s): "%(count_per_frames), count_per_frames/(time.time()-count_time))
                print("FPS (for all): ", frame_count/(time.time()-all_time))
                print("Time all in minutes: %0.1f"%((time.time()-all_time)/60))
                count_time = time.time()

            # input video already in (1280,720)
            frame = cv2.resize(frame, (1280,720))

            # get only entrance from image
            frame = frame[100:400,150:550]

            # model forward
            predicts = detect(model, frame, device)
            
            # process prediction
            frame = process_image(image=frame, predicts=predicts, frame_num=frame_count)

            # cv2.imshow('Video', frame)
            out.write(frame)

            key = cv2.waitKey(1) & 0xFF

            # If the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        else:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
