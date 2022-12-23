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
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

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
def plot_mfcc(path):
    audio, sr = librosa.load(path)
    mfcc_feature = mfcc.mfcc(audio, sr, 0.025, 0.01, 12, nfft = 2205, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    mfcc_feature=np.array(mfcc_feature)
    print(mfcc_feature.shape)
    color = cm.rainbow(np.linspace(0, 1, 13))
    plt.rcParams["figure.figsize"] = [16,12]
    
    for i in range(1,13):
        x_axis=np.arange(0,12)
        y_axis=mfcc_feature[i]  
        plt.plot(x_axis, y_axis, color=color[i], label=f'{i}')

       

    plt.xlabel("MFC coefficients",fontsize=20)
    plt.ylabel("values of the coefficients",fontsize=20)
    
    # fig.write_image('static/assets/images/radar.png')
    plt.legend(bbox_to_anchor=(1.1, 1),loc='upper right', prop={'size': 15})
    plt.savefig('static/assets/images/new_plot.png')
    plt.figure().clear
    
    

def plot_radar (log_likelihood):
    # df = pd.DataFrame(dict(
    # value = log_likelihood,thresohld_value=[.5,.5,.5,.5],
    categories = ['Mohab', 'Marina', 'Omnia','Yousef']
        
    # fig = px.line_polar(df, r = {  }, theta = 'variable', line_close = True, markers = True)
    # fig.update_polars(radialaxis=dict(visible=False,range=[-100,0]))
    # fig.update_traces(fill = 'toself')
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=log_likelihood,
        theta=categories,
        fill='toself',
        name='Product A'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[-30,-30,-30,-30],
        theta=categories,
        fill='toself',
        name='Product B'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[-45, -15]
        )),
    showlegend=False
    )   
    fig.write_image('static/assets/images/radar.png')
    



def plot_radar_speech (log_likelihood_speech):
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
    x=["open","wrong password" ],
    y=np.abs(log_likelihood_speech),
    name='Primary Product',
    marker_color='indianred',
    width=[0.1, 0.1],
    
))
    max = np.max(np.abs(log_likelihood_speech))
    fig.update_layout(yaxis_range=[max-1,max + 1])
    
    fig.write_image('static/assets/images/new_plot.png')
    


    
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
        gmm_files_speech = [ i + '.sav' for i in ['Close', 'Open']]
        
        # model_speech = pickle.load(open('model.pkl', 'rb'))
        # audio_feutures=extract_features_from_input(path)
        # audio_feutures=audio_feutures.reshape(1,65)

        # speech_prediction=model_speech.predict(audio_feutures)
        models    = [pickle.load(open(fname, 'rb') )for fname in gmm_files]
        models_speech=[pickle.load(open(fname, 'rb') )for fname in gmm_files_speech]
        x= extract_features(audio, sr_freq)

        model_scores = np.zeros(len(models))
        log_likelihood = np.zeros(len(models)) 
        for j in range(len(models)):
            gmm = models[j] 
            scores = np.array(gmm.score(x))
            model_scores[j] = scores
            log_likelihood[j] = scores.sum()

        speaker = np.argmax(log_likelihood)

        log_likelihood_speech = np.zeros(len(models_speech)) 
        for j in range(len(models_speech)):
            gmm = models_speech[j] 
            scores = np.array(gmm.score(x))
            log_likelihood_speech[j] = scores.sum()

        speech = np.argmax(log_likelihood_speech)

        flag_speech = False
        flagLst = log_likelihood_speech - max(log_likelihood_speech)
        for i in range(len(flagLst)):
            if flagLst[i] == 0:
                continue
            if abs(flagLst[i]) < 0.8:
                flag_speech = True

        if flag_speech:
            speech = 2

       
        
      
        flag = False
        flagLst = log_likelihood - max(log_likelihood)
        for i in range(len(flagLst)):
            if flagLst[i] == 0:
                continue
            if abs(flagLst[i]) < 0.5:
                flag = True

        if flag:
            speaker = 4



        plot_radar(log_likelihood= log_likelihood)
        plot_radar_speech(log_likelihood_speech = log_likelihood_speech)
        # plot_mfcc(path)
        
        if speech == 0:
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



        

    
   


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

    