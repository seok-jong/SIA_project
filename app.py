from flask import Flask, render_template, request, Response, send_from_directory

import os
import datetime
import re
from io import BytesIO

import tensorflow as tf
import keras
from PIL import Image
import numpy as np
import cv2

os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm


# If you want to use GPU, Comment out this line.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = ({"input_size": 1024,
           "image_size": 928,
           "backbone": "efficientnetb7",
           "method": "unet",
           "class": "mixed"
          })

CLASSES = ['building', 'roads']
orig_path = './origin_images/'
result_path = './result_images/'

if not os.path.exists(orig_path):
    os.mkdir(orig_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)

import albumentations as A

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Resize(config['image_size'], config['image_size'])
    ]
    return A.Compose(_transform)

# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    x *= 255
    return x

# 전처리
BACKBONE = config['backbone']
preprocess_input = sm.get_preprocessing(BACKBONE)
preprocessing = get_preprocessing(preprocess_input)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# 불러올 모델 파일 위치
model.load_weights('./model/mixed_928.h5')

app = Flask(__name__, template_folder="./templates/",
            static_url_path="/images", static_folder="images")

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/healthz", methods=['GET'])
def healthCheck():
    return "", 200


@app.route("/images", methods=['POST'])
def get_result():
    if request.method == "POST":
        width, height = 512, 512
        try:
            curr_time = datetime.datetime.now()
            curr_time = re.sub(r'[ :]','-', str(curr_time))
            source = request.files['source'].read()
            orig = Image.open(BytesIO(source))
            orig.save(f'{orig_path}{curr_time}_orig.png', 'png')
            image = cv2.imread(f'{orig_path}{curr_time}_orig.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            preprocessed = preprocessing(image=image)
            image = preprocessed['image']
            image = np.expand_dims(image, axis=0)
            pr_mask = model.predict(image).round()
            output = denormalize(np.squeeze(pr_mask, axis=0))
            output = cv2.resize(output, (config['input_size'], config['input_size']))
            filename = f'{curr_time}.png'
            cv2.imwrite(result_path + filename, output)
            
            return send_from_directory(result_path, filename, as_attachment=True)

        except Exception as e:
            print("error : %s" % e)
            return Response("fail", status=400)

        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='80')
