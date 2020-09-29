from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__,template_folder='template',static_folder='./template/assets')

# Model saved with Keras model.save()
MODEL_PATH = 'models/trained_model.h5'

#Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    preds = model.predict(img)
    return preds



#####-----------Home Page-----------######
@app.route('/', methods=['GET'])
def root():
    # Main page
    return render_template('index.html')


@app.route('/index.html', methods=['GET'])
def index():
    return render_template('index.html')


#####-----------About Page-----------######
@app.route('/about.html', methods=['GET'])
def about():
    return render_template('about.html')



#####-----------Contact Page-----------######
@app.route('/contact.html', methods=['GET'])
def contact():
    return render_template('contact.html')



#####-----------News Page-----------######
@app.route('/news.html', methods=['GET'])
def news():
    return render_template('news.html')



#####-----------Prevention Page-----------######
@app.route('/prevention.html', methods=['GET'])
def prevention():
    return render_template('prevention.html')



#####-----------Faqs Page-----------######
@app.route('/faqs.html', methods=['GET'])
def faqs():
    return render_template('faqs.html')



#####-----------Upload Page-----------######
@app.route('/upload.html', methods=['GET'])
def uploadq():
    return render_template('upload.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        ##os.remove(file_path)

        
        str1 = 'Covid-19'
        str2 = 'Normal'
        if preds == 1:
            return str1
        else:
            return str2
    return None




if __name__ == '__main__':
        app.run()


