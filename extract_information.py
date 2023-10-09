import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers

from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from tensorflow.keras.models import load_model

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = 'C:\\Users\\ASUS\\PycharmProjects\\text\\transformerocr.pth'

# set device to use cpu
config['device'] = 'cpu'
config['cnn']['pretrained']=True
config['predictor']['beamsearch']=False
detector = Predictor(config)

import cv2
import numpy as np
import os
import shutil

size = 32
best_model = load_model('D:\\data_2\\ocr2.h5')
baseImg = cv2.imread('C:\\Users\\ASUS\\PycharmProjects\\text\\cccd_bezt2_cend.jpg')
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(baseImg, None)

def display_img(cvImg):
    cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,8))
    plt.imshow(cvImg)
    plt.axis('off')
    plt.show()
    
class ImageConstantROI():
    class CCCD(object):
        ROIS = {
            "id": [(362, 236, 925, 296)],
            "name": [(256, 317, 926, 366),],
            "birth": [(522, 362, 742, 407)]
        }

def cropImageRoi(image, roi):
    roi_cropped = image[
        int(roi[1]) : int(roi[3]), int(roi[0]) : int(roi[2])
    ]
    return roi_cropped

def preprocessing_image(img):
    gray = cv2.multiply(img, 1.5)
    blured1 = cv2.medianBlur(gray, 3)
    blured2 = cv2.medianBlur(gray, 51)
    divided = np.ma.divide(blured1, blured2).data
    normed = np.uint8(255*divided/divided.max())
    th, threshed = cv2.threshold(normed, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    
    return threshed


import re

def normalize_name(name):
    first_name = {'trăn': 'trần', 'tràn': 'trần', 'trấn': 'trần', 'trầàn': 'trần', 'pham': 'phạm', 'đó': 'đỗ', 'dỏ': 'đỗ', 'đồ': 'đỗ', 'đồỗ': 'đỗ', 'đỏ': 'đỗ', 'ngộô': 'ngô', 'ngỏ':'ngô', 'hoaàng': 'hoàng', 'nguyên': 'nguyễn', 'nguyễên': 'nguyễn', 'ềiguyển': 'nguyễn', 'nguyêen': 'nguyễn', 'nguyeên': 'nguyễn', 'ta': 'tạ', 'đáng': 'đặng', 'hoaàng': 'hoàng'}
    name = name.lower()
    name = re.sub(r'[^a-zA-Zàáạảãăắằặẳẵâầấậẩẫèéẹẻẽếềệểễêđìíịỉĩòóọỏõôốồộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ /]', '', name)
    name = name.strip()
    names = name.split(' ')
    if names[0].lower() in first_name.keys():
        names[0] = first_name[names[0].lower()]
    return ' '.join(names).title()


def normalize_birth(birth):
    birth = re.sub(r'[^0-9]', '', birth)
    if len(birth) == 8:
        day = birth[:2]
        month = birth[2:4]
        year = birth[4:]
        birth_normalized = day + '/' + month + '/' + year
    else:
        birth_normalized = ' '
    return birth_normalized


def normalize_id(id_number):
    id_number = re.sub(r'[^0-9]', '', id_number)
    
    if len(id_number) != 12:
        id_number = ' '
    return id_number


def extractDataFromIdCard(img, name):
    result = {}
    for key, roi in ImageConstantROI.CCCD.ROIS.items():
        data = ''
        
        for r in roi:
            crop_img = cropImageRoi(img, r)
            
            info = detector.predict(Image.fromarray(crop_img))
            if key == 'name':
                info = normalize_name(info)
            elif key == 'birth':
                info = normalize_birth(info)
                if info == ' ':
                    crop_img = preprocessing_image(crop_img)
                    info = normalize_birth(detector.predict(Image.fromarray(crop_img)))
            else:
                info = normalize_id(info)
                if info == ' ':
                    crop_img = preprocessing_image(crop_img)
                    info = normalize_id(detector.predict(Image.fromarray(crop_img)))
            
            #display_img(crop_img)
            data += info + ' '
            result[key] = data.strip()
        #print(f"{key} : {data.strip()}")
        
    return result


def extract_data(img_path):
    img2 = cv2.imread(img_path)
    img2_nor = cv2.imread(img_path)
    img2_nor = cv2.cvtColor(img2_nor, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des, des1, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.77 * n.distance:
            good_matches.append(m)
    src_points = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(img2, M, (baseImg.shape[1], baseImg.shape[0]))
    result2 = cv2.warpPerspective(img2_nor, M, (baseImg.shape[1], baseImg.shape[0]))
    return extractDataFromIdCard(result, img_path.split('\\')[6]), result2

class ImageConstantROI_for_cnn():
    class CCCD(object):
        ROIS = {
            #"id": [(362, 236, 925, 296)],
            "id": [(362, 236, 710, 290)],
            "name": [(256, 317, 926, 366),],
            #"birth": [(536, 364, 729, 407)]
            "birth": [(522, 362, 742, 407)]
        }

def save_id(img_path, save_folder):
    img2 = cv2.imread(img_path)
    kp1, des1 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des, des1, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.77 * n.distance:
            good_matches.append(m)
    src_points = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(img2, M, (baseImg.shape[1], baseImg.shape[0]))
    for key, roi in ImageConstantROI_for_cnn.CCCD.ROIS.items():
        for r in roi:
            if key == 'id':
                name = img_path.split('\\')[6]
                path = os.path.join(save_folder, name)
                cv2.imwrite(path, cropImageRoi(result, r))

def get_id_img(img_path, key_):
    img2 = cv2.imread(img_path)
    kp1, des1 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des, des1, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.77 * n.distance:
            good_matches.append(m)
    src_points = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(img2, M, (baseImg.shape[1], baseImg.shape[0]))
    for key, roi in ImageConstantROI_for_cnn.CCCD.ROIS.items():
        for r in roi:
            if key == key_:
                return cropImageRoi(result, r)


# Match contours to license plate or character template
def find_contours(dimensions, img, show_pic=True):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(
                intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            if show_pic == True:
                plt.imshow(ii, cmap='gray')
                plt.title('Predict Segments')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

def fix_dimension(img):
    new_img = np.zeros((size,size,3))
    for i in range(3):
        new_img[:,:,i] = img
        return new_img


# Find characters in the resulting images
def segment_characters(image, auto=1, show_pic=False):
    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)

    if auto == 1:
        img_binary_lp = cv2.adaptiveThreshold(img_gray_lp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 9)
    else:
        img_gray_lp = cv2.GaussianBlur(img_gray_lp, (5, 5), 0)
        img_binary_lp = cv2.adaptiveThreshold(img_gray_lp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 15)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH / 6,
                  LP_WIDTH / 2,
                  LP_HEIGHT / 10,
                  2 * LP_HEIGHT / 3]
    #plt.imshow(img_binary_lp, cmap='gray')
    #plt.title('Contour')
    #plt.axis('off')
    #plt.show()
    cv2.imwrite('contour.jpg', img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp, show_pic=show_pic)
    if show_pic == False:
        return char_list
    return None


def predict_cnn(img_path, key_='id'):
    id_img = get_id_img(img_path, key_)
    char = segment_characters(id_img)
    if len(char) != 12:
        char = segment_characters(id_img, auto=0)
    output = []
    for i, ch in enumerate(char):  # iterating over the characters
        ch = 255 - ch
        img_ = cv2.resize(ch, (size, size), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, size, size, 3)  # preparing image for the model
        y = best_model.predict(img, verbose=0)  # predicting the class
        character = np.argmax(y, axis=1)[0]
        output.append(str(character))  # storing the result in a list

    plate_number = ''.join(output)

    return plate_number



