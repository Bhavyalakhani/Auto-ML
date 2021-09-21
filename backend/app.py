# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:28:33 2021

@author: bhavy
"""


from flask import Flask,request,flash,redirect, url_for
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
import os
from flask import Flask,jsonify
from flask_cors import CORS,cross_origin

from AutoML_Script_Classifier1 import runtool
 
UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','xlsx'}


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return 'This is the homepage'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    print("Reached Upload Route")
    target = request.form.get("target")
    print(request.data)
    print(target)
    print(request.files)
    if request.method == 'POST':
        if 'file' not in request.files:
            return "NO file"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "No selected File"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("uploading successful")
            accuracy_val,bestmodel = runtool(os.path.join(app.config['UPLOAD_FOLDER'], filename),target)
            response = jsonify(success=True,output=accuracy_val,model=bestmodel)
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response



if __name__ == "__main__":
    app.debug = True
    app.run()