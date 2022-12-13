from flask import Flask, render_template, request, redirect,jsonify
import speech_recognition as sr
# import Request
import numpy as np
import pickle
import librosa
import os
from scipy.io import wavfile
from flask import jsonify

app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     transcript = ""
#     if request.method == "POST":
#         print("FORM DATA RECEIVED")
#         if "file" not in request.files:
#             return redirect(request.url)
#         file = request.files["file"]
#         if file.filename == "":
#             return redirect(request.url)
#         if file:
#             recognizer = sr.Recognizer()
#             audioFile = sr.AudioFile(file)
#             with audioFile as source:
#                 data = recognizer.record(source)
#             transcript = recognizer.recognize_google(data, key=None)
#     return render_template('index.html', transcript=transcript)

@app.route('/')
def home():
    return render_template('Home2.html')

@app.route("/demo")
def demo():
    return render_template('Demo.html')



# extract featires for a input single audio
def extract_features_from_input(audio):
    extracted_features1 = []
    extracted_features2 = []
    y, sr = librosa.load(audio, mono=True)
    y, index = librosa.effects.trim(y)
#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc_speaker = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 40)
    mfcc_speech = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 60)
    extracted_features1.extend([np.mean(rmse), np.mean(spec_cent),np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)])
    extracted_features2.extend([np.mean(rmse), np.mean(spec_cent),np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)])
    for e in mfcc_speaker:
        extracted_features1.append(np.mean(e))
    for e in mfcc_speech:
        extracted_features2.append(np.mean(e))
    return np.array(extracted_features1), np.array(extracted_features2)
       
       

@app.route('/record', methods=['POST'])
def save():
    
    if request.method == 'POST':
        speaker_model = pickle.load(open('speaker_classifier_final.sav', 'rb'))
        speech_model = pickle.load(open('speech_classifier_final.sav', 'rb'))
        file = request.files['AudioFile']
        # final_features = request.get_json(force=True)
        audio=file.save(os.path.join('static/assets/records/recorded_Sound.wav'))
        path='static/assets/records/recorded_Sound.wav'
        # sr, audio = wavfile.read(
        #     'static/assets/records/recorded_Sound.wav')
        # if len(audio.shape) > 1:
        #     audio = audio[:, 0]
        speakr_final_features, speech_final_featurs = extract_features_from_input(path)
        speakr_final_features, speech_final_featurs = speakr_final_features.reshape(1,45), speech_final_featurs.reshape(1, 65)
        # final_features2=audio_features.reshape(1,65)
        speaker_prediction = speaker_model.predict(speakr_final_features)
        speech_prediction = speech_model.predict(speech_final_featurs)
        speakers_list = [(0, 'Marina'), (1, 'Mohab'), (2, 'Yousef'), (3, 'Omnia'),(4,'others')]
        speech_list=[0,1]
        for iterator, speaker in enumerate(speakers_list):
            if speaker[0] == speaker_prediction[0]:
                predict=speaker[1]
                if predict=='others':
                    return jsonify({'output' :"unregisterd speaker, please try again"})
                    
                elif(speech_list[0]==speech_prediction[0]):
                        
                    return jsonify({'output' :f" Correct password, welcome { predict }"})
                else:
                    return jsonify({'output' :"Wrong password, please try again"})





if __name__ == "__main__":
    app.run(debug=True, threaded=True)

    