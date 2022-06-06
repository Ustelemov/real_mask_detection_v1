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
import tensorflow_hub as hub
import tensorflow as tf

import urllib.request

import uuid

from time import strftime
from time import gmtime

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, strip_optimizer, set_logging
from utils.torch_utils import select_device, load_classifier, time_synchronized

from sort import Sort
import urllib 
import requests

class FaceDetector:
    def __init__(self, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model = attempt_load(weights, map_location=self.device)
        self.img_size = 320
        self.conf_thres = 0.2
        self.iou_thres = 0.2
   
    def __preprocess_image(self, orgimg):
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        
        if r != 1:  
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x320x320

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
 
    def __postprocess_prediction(self,orgimg,img,pred):
        h0, w0 = orgimg.shape[:2]  # orig hw
        ret = []
        # Add 40% of width\height padding for face
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(self.device)  # normalization gain whwh
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

                    x_top_left = int(x_top_left - width*0.4)
                    y_top_left = int(y_top_left - height*0.4)
                    x_bottom_right = int(x_bottom_right + width*0.4)
                    y_bottom_right = int(y_bottom_right + height*0.4)

                    x_top_left = 0 if x_top_left < 0 else x_top_left
                    y_top_left = 0 if y_top_left < 0 else y_top_left
                    x_bottom_right = w0 if x_bottom_right > w0 else x_bottom_right
                    y_bottom_right = h0 if y_bottom_right > h0 else y_bottom_right

                    ret.append([x_top_left, y_top_left, x_bottom_right, y_bottom_right, conf])
        return ret

    def detect_face_boxes(self, orgimg):
        img = self.__preprocess_image(orgimg)

        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

        ret = self.__postprocess_prediction(orgimg,img,pred)

        return ret


class MaskClassificator:
    def __init__(self, model_dir, img_size):
       self.model = hub.KerasLayer(model_dir)
       self.img_size = img_size 

    def __preprocess_image(self, orgimg):
        img0 = copy.deepcopy(orgimg)
        img0 = img0[...,::-1]
        img0 = np.array(img0)
        # reshape into shape [batch_size, height, width, num_channels]
        img_reshaped = tf.reshape(img0, [1, img0.shape[0], img0.shape[1], img0.shape[2]])
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img_reshaped = tf.image.convert_image_dtype(img_reshaped, tf.float32)
        img_reshaped = tf.image.resize(img_reshaped, [self.img_size, self.img_size])
        return img_reshaped

    def proccess_image_classification(self, orgimg):
        image = self.__preprocess_image(orgimg)

        # Run model on image
        logits = self.model(image)

        return tf.nn.softmax(logits)[0]


class ImageProcessor:
    def __init__(self, input_FPS, detector_weights, classification_model_dir, classification_model_img_size, is_stream):
        self.face_detector = FaceDetector(weights=detector_weights)
        self.mask_classificator = MaskClassificator(model_dir=classification_model_dir, img_size=classification_model_img_size)
        self.tracker = Sort(max_age=30, min_hits=0)

        self.faces = dict()

        self.input_FPS = input_FPS
        self.is_stream = is_stream
        
        self.score_threshold = 0.8
        self.count_threshold = 4
        self.face_size_threshold = 35

        self.save_path = 'output'
        mkdir(self.save_path)
        self.saved_count=0

        self.telegram_token = '5477208331:AAEwzfPttyHNpq1tJIiYa68BHmVNtpEfLrk'
        self.telegram_channeld_id = -1001438367642


    def __add(self, id, arr, score, image_num):
        if score < self.score_threshold:
            return

        if id not in self.faces:
            self.faces[id] = []
        self.faces[id].append((arr,score, image_num))
    
    def __remove(self, id):
        if id not in self.faces:
            return
        
        dataArr = self.faces[id]

        # remove element
        del self.faces[id]

        if len(dataArr) < self.count_threshold:
            return

        self.__save(dataArr)

    def __save(self,dataArr):
        
        # select best face by size
        img = dataArr[0][0]
        image_num = dataArr[0][2]

        for data in dataArr[:]:
            if img.shape[0] * img.shape[1] < data[0].shape[0] * data[0].shape[1]:
                img = data[0]
                image_num = data[2] 
            
        # classification
        class_probs = self.mask_classificator.proccess_image_classification(img)

        # save local
        if self.is_stream:
            image_time = strftime("%H:%M:%S", gmtime())   
        else:
            image_time = strftime("%H:%M:%S", gmtime(int(image_num/self.input_FPS)))

        id = str(uuid.uuid1())
        image_path = "{0}/{1}_{2}.jpg".format(self.save_path, image_time,id)

        cv2.imwrite(image_path, img)

        caption = 'Mask - {0:.2f} Nomask - {1:.2f}'.format(class_probs[0], class_probs[1])
        self.__send_image_telegram(image_path=image_path, caption=caption)

        self.saved_count += 1
        print("---------saved count: %s---------"%(self.saved_count))


    def __send_image_telegram(self,image_path, caption):
        files = {
            'document': open(image_path, 'rb'),
        }
        
        url = 'https://api.telegram.org/bot{0}/sendDocument?chat_id={1}&caption={2}'.format(
            self.telegram_token, self.telegram_channeld_id, caption)

        response = requests.post(url, files=files)
        if response.status_code != 200:
            print('Bad response code while send to telegram: ', response.status_code)


    def process_image(self, image, image_num):
        # model forward
        predicts = self.face_detector.detect_face_boxes(image)

        h,w,c = image.shape

        result_image = image.copy()
        faces = []
        for predict in predicts:
            x1, y1, x2, y2, score = predict     

            if x2-x1 > self.face_size_threshold and y2-y1 > self.face_size_threshold:
                faces.append(predict)

        trackers, removed = self.tracker.update(np.array(faces))
        
        # visualize and add
        for d in trackers:
            score = d[5]
            d = d.astype(np.int32)

            # add after check for bounds
            x1 = 0 if d[0] < 0 else d[0]
            y1 = 0 if d[1] < 0 else d[1]
            x2 = w if d[2] > w else d[2]
            y2 = h if d[3] > h else d[3]


            face = image[y1:y2, x1:x2]
            self.__add(d[4],face, score,image_num)

        # process removed
        for rm in removed:
            self.__remove(rm)

def mkdir(path):
    path.strip()
    path.rstrip('\\')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def get_image_from_capture(cap):
    ret,image = cap.read()
    return ret,image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to mp4 file or MJPEG stream")
    parser.add_argument("--detector_weights", type=str, 
        help="path to face detector weights", default='./weights/yolov5s-face.pt')
    parser.add_argument("--classification_model_dir", type=str,
        help="path to classification model dir", default='./weights/visitors_model')
    parser.add_argument("--classification_model_img_size", type=int,
        help="weight and height of input image for classificaion (size of input layer)", default=128)

    args = parser.parse_args()

    # if it's url
    if urllib.parse.urlparse(args.input).scheme != "":
        is_stream = True
    else:
        is_stream = False
    
    cap = cv2.VideoCapture(args.input)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_FPS = int(cap.get(cv2.CAP_PROP_FPS))
    print("Input video: %s FPS: %d Length frames: %d"%(args.input, input_FPS, length))

    image_processor = ImageProcessor(input_FPS=input_FPS, detector_weights=args.detector_weights, 
    classification_model_dir=args.classification_model_dir, 
    classification_model_img_size=args.classification_model_img_size, is_stream=is_stream)

    image_count = 0
    count_time = time.time()
    all_time = time.time()
    count_per_images = 1000

    while True:
        ret,image = get_image_from_capture(cap)
        if ret:
            #print
            image_count = image_count + 1
            if image_count%count_per_images == 0:
                print("---------------------------------------------------------------------------------")
                print("----------------------------", image_count," of ", length, "-----------------------------")
                print("---------------------------------------------------------------------------------")
                print("FPS (for %s): "%(count_per_images), count_per_images/(time.time()-count_time))
                print("FPS (for all): ", image_count/(time.time()-all_time))
                print("Time all in minutes: %0.1f"%((time.time()-all_time)/60))
                count_time = time.time()

            # input video already in (1280,720)
            image = cv2.resize(image, (1280,720))

            # get only entrance from image
            image = image[100:400,150:550]

            # process image
            image_processor.process_image(image,image_count)

            # cv2.imshow('Video', image)

            key = cv2.waitKey(1) & 0xFF

            # If the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()