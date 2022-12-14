import os
import librosa
import numpy as np
import pickle

def get_statistics(list: list):
    '''
    Input: a list of values
    Output: a list with the minimum value, maximum value, mean and standard deviation of the input
    '''
    list = np.array(list)
    return [min(list), max(list), np.mean(list), np.std(list)]

def get_names(pitch: bool = 1, dynamics: bool = 1, rhythm: bool = 1, timbre: bool = 1):
    '''
    Input: booleans indicating the types of features to be used (pitch, dynamics, rhythm and timbre)
    Output: Returns a list of the names of the features for the activated types of features
    '''
    pitch = []
    for i in range(12):
        pitch.extend([str(i+1)+"_pitch_min", str(i+1)+"_pitch_max", str(i+1)+"_pitch_mean", str(i+1)+"_pitch_std"])
    
    dynamics = []
    dynamics.extend(["spectral_centroid_min", "spectral_centroid_max", "spectral_centroid_mean", "spectral_centroid_std"])
    dynamics.extend(["zero_crossing_rate_min", "zero_crossing_rate_max", "zero_crossing_rate_mean", "zero_crossing_rate_std"])
    dynamics.extend(["spectral_rolloff_min", "spectral_rolloff_max", "spectral_rolloff_mean", "spectral_rolloff_std"])

    rhythm = []
    rhythm.extend(["tempogram_min", "tempogram_max", "tempogram_mean", "tempogram_std"])

    timbre = []
    for i in range(20):
        timbre.extend([str(i+1)+"_mfcc_min", str(i+1)+"_mfcc_max", str(i+1)+"_mfcc_mean", str(i+1)+"_mfcc_std"])

    features_names = []
    if(pitch):
        features_names.extend(pitch)
    if(dynamics):
        features_names.extend(dynamics)
    if(rhythm):
        features_names.extend(rhythm)
    if(timbre):
        features_names.extend(timbre)

    return features_names
    
def feature_extraction(path: str, pitch: bool = 1, dynamics: bool = 1, rhythm: bool = 1, timbre: bool = 1):
    '''
    Input:
    - path: the path of the song
    - pitch: 1 to extract pitch features, 0 to not extract
    - dynamics: 1 to extract dynamics features, 0 to not extract
    - rhythm: 1 to extract rhythm features, 0 to not extract
    - timbre: 1 to extract timbre features, 0 to not extract

    Output:
    - A list of lists with all the statistics for the selected features
    '''
    y, sr = librosa.load(path)
    features = []

    if(pitch):
    # pitch features are chroma
        chroma_stft = librosa.feature.chroma_stft(y)
        for i in range(len(chroma_stft)):
            features.extend(get_statistics(chroma_stft[i]))

    if(dynamics):
    # dynamical features
        centroid  = librosa.feature.spectral_centroid(y, sr)[0]
        features.extend(get_statistics(centroid))
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, pad=False)[0]
        features.extend(get_statistics(zero_crossing_rate))
        spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)[0]
        features.extend(get_statistics(spectral_rolloff))

    if(rhythm):
    #rhythm features
        tempogram = librosa.feature.tempogram(y, sr)
        print(len(tempogram))
        print(len(tempogram[0]))
        print(len(tempogram[1]))
        print(len(tempogram[2]))
        #features.extend(get_statistics(tempogram))
    
    if(timbre):
    #timbre features
        mfccs = librosa.feature.mfcc(y, sr=sr)
        for i in range(len(mfccs)):
            features.extend(get_statistics(mfccs[i]))
        print(len(mfccs[1]))

    return features


# Read the dataset and extract the features
names = get_names()
class_ = ["Data/Like/" , "Data/Dislike/"]
features = []
label = 0
for cla in class_:
    for fil in os.listdir(cla):
        print(fil)
        feat = feature_extraction(cla+fil)
        feat.append(label)
        features.append(feat)
    label += 1

f = open('dataset_features.pckl', 'wb')
pickle.dump(np.array(features), f)
f.close()