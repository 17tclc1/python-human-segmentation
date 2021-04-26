from flask import Flask, render_template, redirect, request, jsonify,Blueprint
from flask_jwt_extended import JWTManager, jwt_required, create_access_token,get_jwt_identity
from flask_cors import CORS
import datetime
#from apscheduler.schedulers.background import BackgroundScheduler
import cv2, time, threading, os,pickle
import numpy as np
import glob
import shutil
from slugify import slugify
import base64
# path
path_faceCascade = cv2.CascadeClassifier('./training/face_detection_classifier/intel_frontal_face_classifier.xml')
path_embedding_model = './training/openface_nn4.small2.v1.t7'
path_recognizer = './training/output/recognizer.pickle'
path_le_pickle = './training/output/le.pickle'
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'secretkey51'  # Change this!
jwt = JWTManager(app)
CORS(app)
# Server IP Address
server = 'localhost'