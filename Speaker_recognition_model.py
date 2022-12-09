# The output of this section is the CSV files with the data to be handle by the model
from sklearn.preprocessing import StandardScaler
import pickle
import zipfile as zf
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import os
import csv
import zipfile
import librosa
import pandas as pd
from matplotlib import cm
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
CREATE_CSV_FILES = True

# Defines the names of the CSV files
TRAIN_CSV_FILE = "Speaker_Train_File.csv"
TEST_CSV_FILE = "Speaker_Test_File.csv"


files = zf.ZipFile("recordings/Spea.zip", 'r')
files.extractall('Audio_Data')
files.close()


def extractWavFeatures(soundFilesFolder, csvFileName):
    print("The features of the files in the folder " +
          soundFilesFolder+" will be saved to "+csvFileName)
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 41):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    print('CSV Header: ', header)
    file = open(csvFileName, 'w', newline='')
    # with file:
    writer = csv.writer(file)
    writer.writerow(header)
    for filename in os.listdir(soundFilesFolder):
        if filename.endswith('.wav'):
            number = f'{soundFilesFolder}/{filename}'
            y, sr = librosa.load(number, mono=True, duration=30)
            # remove leading and trailing silence
            y, index = librosa.effects.trim(y)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rms(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            writer.writerow(to_append.split())
    file.close()
    print("End of extractWavFeatures")


if (CREATE_CSV_FILES == True):
    extractWavFeatures(
        "Audio_Data/Speaker_recognition_train_data", TRAIN_CSV_FILE)
    extractWavFeatures(
        "Audio_Data/Speaker_recognition_test_data", TEST_CSV_FILE)
    print("CSV files are created")
else:
    print("CSV files creation is skipped")

# Reading a dataset and convert file name to corresbonding umnber


def preProcessData(csvFileName):
    header_name_list = ['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14',
                        'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'mfcc21', 'mfcc22', 'mfcc23', 'mfcc24', 'mfcc25', 'mfcc26', 'mfcc27', 'mfcc28', 'mfcc29', 'mfcc30', 'mfcc31', 'mfcc32', 'mfcc33', 'mfcc34', 'mfcc35', 'mfcc36', 'mfcc37', 'mfcc38', 'mfcc39', 'mfcc40', 'label']
    print(csvFileName + " will be preprocessed")
    data = pd.read_csv(csvFileName)
    # we have 4 speakers:
    # 0: Marina
    # 1: Mohab
    # 2: Omnia
    # 3: Youssef
    filenameArray = data['filename']
    speakerArray = []
    for filename in filenameArray:
        if "marina" in filename:
            speaker = 0
        elif "mohab" in filename:
            speaker = 1
#          elif "omnia" in filename:
#                speaker = "2"
        elif "Youssef" in filename:
            speaker = "3"
        else:
            speaker = "4"
        speakerArray.append(speaker)
    data['number'] = speakerArray
    # Dropping unnecessary columns
    data = data.drop(['filename'], axis=1)
    data = data.drop(['label'], axis=1)
    data = data.drop(['chroma_stft'], axis=1)
    return data


trainData = preProcessData(TRAIN_CSV_FILE)
testData = preProcessData(TEST_CSV_FILE)

# Splitting the dataset into training, validation and testing dataset
X = np.array(trainData.iloc[:, :-1], dtype=float)
y = trainData.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_test = np.array(testData.iloc[:, :-1], dtype=float)
y_test = testData.iloc[:, -1]

# Normalizing the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Create an object (model)
dtr1 = DecisionTreeClassifier()

# Fit (train) the model
dtr1.fit(X_train, y_train)
# print("model is fit successfully")
# dumping the model into a .pkl file
pickle.dump(dtr1, open('finalized_model.sav', 'wb'))
