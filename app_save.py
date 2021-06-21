from flask import Flask, render_template, request, Response, send_from_directory

import os
import datetime
import re
from io import BytesIO

import tensorflow as tf
from PIL import Image
import numpy as np

from image_preprocessor import preprocess, denormalize
from auto_painter import load_auto_painter_model, generate_image

# If you want to use GPU, Comment out this line.
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

model = load_auto_painter_model()
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
            orig_path = './origin_images/'
            orig.save(f'{orig_path}{curr_time}_orig.png', 'png')
            adjusted_image = preprocess(source, width, height)
            image = denormalize(adjusted_image)
            result = generate_image(model, adjusted_image)
            image = Image.fromarray(result)
            image_path = './result_images/'
            filename = f'{curr_time}.png'
            image.save(image_path + filename)
            
            return send_from_directory(image_path, filename, as_attachment=True)

        except Exception as e:
            print("error : %s" % e)
            return Response("fail", status=400)

        
if __name__ == '__main__':
    app.run(host='0.0.0.0')
