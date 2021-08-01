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
from flask import Flask
from flask_cors import CORS, cross_origin

from AutoML_Script_Classifier1 import runtool
 
UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','xlsx'}


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

cors = CORS(app)

@app.route("/")
@cross_origin()
def index():
    return 'This is the homepage'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           

@app.route('/upload', methods=[ 'POST'])
@cross_origin()
def upload_file():
    target = request.form.get("target")
    print(target)
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
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
            accuracy_val = runtool(os.path.join(app.config['UPLOAD_FOLDER'], filename),target)
            return str(accuracy_val)



if __name__ == "__main__":
    app.run(debug=True,threaded=True)