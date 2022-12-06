from flask import Flask, render_template, request
import os
from scipy.io import wavfile

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/record', methods=['POST'])
def save():
    if request.method == 'POST':
        file = request.files['AudioFile']

        file.save(os.path.join(
            'static/assests/records/recorded_Sound.wav'))
        # sr, audio = wavfile.read(
        #     'static/assets/records/assets/records/recordedAudio.wav')
        # if len(audio.shape) > 1:
        #     audio = audio[:, 0]

    return[]


if __name__ == '__main__':
    app.run(debug=True)
