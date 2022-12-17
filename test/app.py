import os
import uuid
from flask import Flask, flash, request, redirect,url_for,render_template, jsonify
import wave
from scipy.io.wavfile import write
import librosa as lr 
from werkzeug.utils import secure_filename
import joblib
import numpy as np
import extract_feature as ef
import python_speech_features as mfcc
from sklearn import preprocessing
import pickle

app = Flask(__name__)

# UPLOAD_FOLDER = 'static/file/'
app.secret_key = "cairocoders-ednalan"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def calculate_delta(array):
   
    rows, cols = array.shape
    # print(rows)
    # print(cols)
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first =0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    global combined  
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft = 2205, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
#     print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta)) 
    return combined

def predict_person(file_path):
    audio, sr_freq = lr.load(file_path)
    S = np.abs(lr.stft(audio))

    gmm_files = [ i + '.joblib' for i in ['amira', 'ezzat', 'mariam', 'osama']]
    models    = [joblib.load(fname) for fname in gmm_files]
    x= extract_features(audio, sr_freq)

    
    log_likelihood = np.zeros(len(models)) 
    for j in range(len(models)):
        gmm = models[j] 
        scores = np.array(gmm.score(x))
        log_likelihood[j] = scores.sum()

    winner = np.argmax(log_likelihood)
    print(log_likelihood)


    flag = False
    flagLst = log_likelihood - max(log_likelihood)
    for i in range(len(flagLst)):
        if flagLst[i] == 0:
            continue
        if abs(flagLst[i]) < 0.4:
            flag = True


    print(flagLst)
    print(winner)
    if flag:
        winner = 4

    print(winner)

    if winner == 0:
        return "amira"
    elif winner ==1:
        return "ezzat"
    elif winner ==2:
        return "mariam"   
    elif winner ==3:
        return "osama" 
    else: 
        return "others" 




def predict_scentence(file_path):
    audio, sr_freq = lr.load(file_path)
    S = np.abs(lr.stft(audio))

    gmm_files = [ i + '.joblib' for i in ['closedoor','closelaptop','openbook','opendoor']]
    models    = [joblib.load(fname) for fname in gmm_files]
    x= extract_features(audio, sr_freq)

    
    log_likelihood = np.zeros(len(models)) 
    for j in range(len(models)):
        gmm = models[j] 
        scores = np.array(gmm.score(x))
        log_likelihood[j] = scores.sum()

    winner = np.argmax(log_likelihood)
    print(log_likelihood)


    flag = False
    flagLst = log_likelihood - max(log_likelihood)
    for i in range(len(flagLst)):
        if flagLst[i] == 0:
            continue
        if abs(flagLst[i]) < 0.4:
            flag = True

    if flag:
        winner = 5

    print(winner)

    if winner == 0:
        return "close door"
    elif winner ==1:
        return "close laptop"
    elif winner ==2:
        return "open book"   
    elif winner ==3:
        return "open door"   
    else:
        return "other" 
    
    





@app.route('/')
def root():
    return render_template('index.html') 

 
@app.route('/predict', methods=['POST'])
def save_record():
    if request.files['file']:
        file = request.files['file'] 
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(file.filename))
        file_path += '.wav'
        file.save(file_path)
        return jsonify({
                    'person':predict_person(file_path),
                    'sentence':predict_scentence(file_path)
                })
    return 400

if __name__ == '__main__':
    app.run(debug=False)
   