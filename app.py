from flask import Flask, render_template, request, redirect,jsonify
import speech_recognition as sr
# import Request
import numpy as np
import pickle
import librosa 
import os
from scipy.io import wavfile
from flask import jsonify
import python_speech_features as mfcc
from sklearn import preprocessing

app = Flask(__name__)

def calculate_delta(array):
   
    rows, cols = array.shape
 
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
def extract_features_from_input(input_audio):
    extracted_features = []
    y, sr = librosa.load(input_audio, mono=True)
    y, index = librosa.effects.trim(y)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 60)
    extracted_features.extend([np.mean(rmse),  np.mean(spec_cent),np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)])
    for e in mfcc:
        extracted_features.append(np.mean(e))
    return np.array(extracted_features)
def extract_features(audio,rate):
    global combined  
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft = 2205, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta)) 
    return combined
@app.route('/')
def home():
    return render_template('main.html')



       

@app.route('/record', methods=['POST'])
def save():
    
    if request.method == 'POST':
    
        
        file = request.files['AudioFile']
        audio=file.save(os.path.join('static/assets/records/recorded_Sound.wav'))
        path='static/assets/records/recorded_Sound.wav'
        audio, sr_freq = librosa.load(path)

        gmm_files = [ i + '.sav' for i in ['Mohab', 'Marina', 'Omnia','Yousef']]
        # gmm_files_speech = [ i + '.sav' for i in ['Close', 'Open']]
        
        model_speech = pickle.load(open('model.pkl', 'rb'))
        audio_feutures=extract_features_from_input(path)
        print(audio_feutures.shape)
        audio_feutures=audio_feutures.reshape(1,65)

        speech_prediction=model_speech.predict(audio_feutures)
        models    = [pickle.load(open(fname, 'rb') )for fname in gmm_files]
        x= extract_features(audio, sr_freq)

        
        log_likelihood = np.zeros(len(models)) 
        for j in range(len(models)):
            gmm = models[j] 
            scores = np.array(gmm.score(x))
            log_likelihood[j] = scores.sum()

        speaker = np.argmax(log_likelihood)

        # log_likelihood_speech = np.zeros(len(models_speech)) 
        # for j in range(len(models_speech)):
        #     gmm = models_speech[j] 
        #     scores = np.array(gmm.score(x))
        #     log_likelihood_speech[j] = scores.sum()

        # speech = np.argmax(log_likelihood_speech)

        # flag_speech = False
        # flagLst = log_likelihood_speech - max(log_likelihood_speech)
        # for i in range(len(flagLst)):
        #     if flagLst[i] == 0:
        #         continue
        #     if abs(flagLst[i]) < 0.8:
        #         flag_speech = True

        # if flag_speech:
        #     speech = 2

       
        
      
        flag = False
        flagLst = log_likelihood - max(log_likelihood)
        for i in range(len(flagLst)):
            if flagLst[i] == 0:
                continue
            if abs(flagLst[i]) < 0.5:
                flag = True

        if flag:
            speaker = 4

        
        if speech_prediction[0] == 1:
            return jsonify({'output' :"Wrong password, try again."}) 
       
        elif speaker == 0:

            return jsonify({'output' :"Mohab!"})
        elif speaker ==1:
            return jsonify({'output' :"Marina"})
        elif speaker ==2:
            return jsonify({'output' :"Omnia"})
        elif speaker ==3:
            return jsonify({'output' :"Yousef"})  
        
        else: 
            return jsonify({'output' :"Unregistered speaker"}) 

        # if speaker == 0:

        #     return jsonify({'output' :"Correct password, welcome Mohab!"})
        # elif speaker ==1:
        #     return jsonify({'output' :"Correct password, welcome Marina"})
        # elif speaker ==2:
        #     return jsonify({'output' :"Correct password, welcome Omnia!"})
        # elif speaker ==3:
        #     return jsonify({'output' :"Correct password, welcome Yousef!"})  
        
        # else: 
        #     return jsonify({'output' :"Unregistered speaker, try again."}) 
        
        





if __name__ == "__main__":
    app.run(debug=True, threaded=True)

    