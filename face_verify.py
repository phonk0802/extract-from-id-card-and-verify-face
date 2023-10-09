import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import math 
import cv2
import re
from tqdm import tqdm

sns.set_style('dark')


# In[2]:


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image

model = InceptionResnetV1(pretrained='vggface2').eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

phonk_base = 'D:\\3_person_face_dataset\\base\\phonk\\cccd_me.jpg'
mom_base = 'D:\\3_person_face_dataset\\base\\mom\\cccd_mom.jpg'
minh_base = 'D:\\3_person_face_dataset\\base\\minh\\cccd_minh.jpg'
phuong_base = 'D:\\3_person_face_dataset\\base\\phuong\\cccd_phuong.jpg'
hoang_base = 'D:\\3_person_face_dataset\\base\\hoang\\cccd_hoang.jpg'
trung_base = 'D:\\3_person_face_dataset\\base\\trung\\cccd_trung.jpg'

base = {'phonk': phonk_base, 'mom': mom_base, 'minh': minh_base, 'phuong': phuong_base, 'hoang': hoang_base, 'trung': trung_base}

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small, mobilenet_v2, mobilenet_v3
from keras import models
from keras.applications.vgg16 import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
conv_base.trainable=False
model_mask = load_model('E:\\Downloads\\model1.h5')


def extract_face(img_path):
    mtcnn_post = MTCNN(
        image_size=170, margin=3, min_face_size=15,
        thresholds=[0.6, 0.7, 0.7], factor=0.695, post_process=True,
        device=device
    )

    face = Image.open(img_path)
    face_for_verify = mtcnn_post(face, return_prob=True)[0]
    face_for_mask = face_for_verify * 128 + 127.5
    face_for_mask = F.interpolate(face_for_mask.unsqueeze(0), size=(224, 224), mode="bicubic").squeeze(0)
    # face_for_mask = mtcnn_impost(face, return_prob=True)[0]
    # face_ = F.interpolate(face_.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    # face_ = face_.squeeze(0)
    # face_ = np.reshape(face_, (3, 224, 224))
    box, prob, landmark = mtcnn_post.detect(face, landmarks=True)

    return face_for_verify, face_for_mask, prob, landmark

def check_mask(face):
    face = np.transpose(face, (1, 2, 0))
    face = np.expand_dims(face, axis=0)
    fe = conv_base.predict(face, verbose=0)
    fe = np.reshape(fe, (1, 7*7*512))
    pred = model_mask.predict(fe, verbose=0)
    
    if pred > 0.01:
        return True
    else:
        return False

def get_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b) 
    
    cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def visualize(image, landmarks_, angle_R_, angle_L_, pred_):
    fig , ax = plt.subplots(1, 1, figsize= (10,10))
    ax.set_title(pred_[0])
    for landmarks, angle_R, angle_L, pred in zip(landmarks_, angle_R_, angle_L_, pred_):
        
        if pred == 'Frontal':
            color = 'white'
        elif pred == 'Right Profile':
            color = 'blue'
        else:
            color = 'red'
            
        point1 = [landmarks[0][0], landmarks[1][0]]
        point2 = [landmarks[0][1], landmarks[1][1]]

        point3 = [landmarks[2][0], landmarks[0][0]]
        point4 = [landmarks[2][1], landmarks[0][1]]

        point5 = [landmarks[2][0], landmarks[1][0]]
        point6 = [landmarks[2][1], landmarks[1][1]]
        for land in landmarks:
            ax.scatter(land[0], land[1])
        plt.plot(point1, point2, 'y', linewidth=3)
        plt.plot(point3, point4, 'y', linewidth=3)
        plt.plot(point5, point6, 'y', linewidth=3)
        plt.text(point1[0], point2[0], f"{pred} \n {math.floor(angle_L)}, {math.floor(angle_R)}", 
                size=20, ha="center", va="center", color=color)
    ax.axis('off')
    ax.imshow(image)

def check_angle(im, face, probs, landmarks, show_pic=True):
    angle_rs = []
    angle_ls = []
    preds = []
    face = np.transpose(face, (1, 2, 0))
    for box, landmark, prob in zip(face, landmarks, probs):
        if prob > 0.9:
            angle_r = get_angle(landmark[0], landmark[1], landmark[2])
            angle_l = get_angle(landmark[1], landmark[0], landmark[2])
            
            angle_rs.append(angle_r)
            angle_ls.append(angle_l)
            
            if abs(int(angle_r) - int(angle_l)) < 23:
                pred = 'Frontal'
                preds.append(pred)
            else:
                if angle_r < angle_l:
                    pred = 'Left Profile'
                else:
                    pred = 'Right Profile'
                preds.append(pred)
        else:
            print('The detected face is Less then the detection threshold')
    if show_pic == True:
        visualize(im, landmarks, angle_rs, angle_ls, preds)
    return pred


def face_by_face_verify(img_path1, img_path2, threshold=0.888, show_pic=False):
    try:
        face1, _1, prob1, landmark1 = extract_face(img_path1)
        face2, _2, prob2, landmark2 = extract_face(img_path2)
        img2 = Image.open(img_path2)
        if check_mask(_2):
            im = cv2.imread(img_path2)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            plt.imshow(im)
            plt.axis('off')
            plt.title('Mask')
            plt.show()
            return 'Mask detected', -1
        if check_angle(img2, face2, prob2, landmark2, show_pic=show_pic) != 'Frontal':
            angle = check_angle(img2, face2, prob2, landmark2, show_pic=True)
            return angle + ' detected', -1
        
        face1_emb = model(face1.unsqueeze(0)).detach().cpu()
        face2_emb = model(face2.unsqueeze(0)).detach().cpu()

        dist = (face1_emb - face2_emb).norm().item()

        if dist > threshold:
            return False, dist
        return True, dist
    except AttributeError:
        return 'Error'
        pass



def face_verify(img_path, person, threshold=0.888, show_pic=True):
    if person not in base.keys():
        return 'This person is not in the database'

    try:
        face1, _1, prob1, landmark1 = extract_face(base[person])
        face2, _2, prob2, landmark2 = extract_face(img_path)

        img2 = Image.open(img_path)

        if check_mask(_2):
            return 'Mask detected', -1
        if check_angle(img2, face2, prob2, landmark2, show_pic=show_pic) != 'Frontal':
            return 'Side Profile detected', -1

        face1_emb = model(face1.unsqueeze(0)).detach().cpu()
        face2_emb = model(face2.unsqueeze(0)).detach().cpu()

        dist = (face1_emb - face2_emb).norm().item()

        if dist > threshold:
            return False, dist
        return True, dist

    except AttributeError:
        return 'Error'
        pass