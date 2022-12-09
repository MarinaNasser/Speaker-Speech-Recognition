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
    return render_template('Home1.html')

@app.route("/demo")
def demo():
    return render_template('Demo.html')



# extract featires for a input single audio
def extract_features_from_input(audio):
    extracted_features = []
    y, sr = librosa.load(audio, mono=True)
    y, index = librosa.effects.trim(y)
#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 40)
    extracted_features.extend([np.mean(rmse), np.mean(spec_cent),np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)])
    for e in mfcc:
        extracted_features.append(np.mean(e))
    return np.array(extracted_features)
       
       

@app.route('/record', methods=['POST'])
def save():
    
    if request.method == 'POST':
        speaker_model = pickle.load(open('finalized_model.sav', 'rb'))
        
        file = request.files['AudioFile']
        # final_features = request.get_json(force=True)

        audio=file.save(os.path.join(
            'static/assets/records/recorded_Sound.wav'))
        path='static/assets/records/recorded_Sound.wav'
        # sr, audio = wavfile.read(
        #     'static/assets/records/recorded_Sound.wav')
        # if len(audio.shape) > 1:
        #     audio = audio[:, 0]
        audio_features = extract_features_from_input(path)
        final_features = audio_features.reshape(1,45)
        prediction = speaker_model.predict(final_features)
        speakers_list = [(0, 'Marina'), (1, 'Mohab'), (3, 'Yousef'), (4, 'Others')]
        for iterator, speaker in enumerate(speakers_list):
            if speaker[0] == prediction[0]:

                predict=speaker[1]
                return jsonify({'output' :predict})






if __name__ == "__main__":
    app.run(debug=True, threaded=True)

    