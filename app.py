from flask import Flask, render_template, request, redirect,jsonify
import speech_recognition as sr
# import Request
import numpy as np
import pickle
import librosa
import os
app = Flask(__name__)
speaker_model = pickle.load(open('finalized_model.sav', 'rb'))

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
    return render_template('Home.html')

@app.route("/demo")
def demo():
    return render_template('Demo.html')



# extract featires for a input single audio
def extract_features_from_input(input_audio):
    extracted_features = []
    y, sr = librosa.load(input_audio, mono=True, duration=30)
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
        file = request.files['AudioFile']

        file.save(os.path.join(
            'static/assets/records/recorded_Sound.wav'))
        # sr, audio = wavfile.read(
        #     'static/assets/records/assets/records/recordedAudio.wav')
        # if len(audio.shape) > 1:
        #     audio = audio[:, 0]

    return[]
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    audio_features = extract_features_from_input(request.files["file"])
    final_features = audio_features.reshape(1,45)
    prediction = speaker_model.predict(final_features)
    speakers_list = [(0, 'Marina'), (1, 'Mohab'), (3, 'Yousef'), (4, 'Others')]
    for iterator, speaker in enumerate(speakers_list):
        if prediction[0] == speaker[0]:
           return render_template('index.html', prediction_text='Predicted Speaker $ {}'.format(prediction[1]))



if __name__ == "__main__":
    app.run(debug=True, threaded=True)

    