import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm, metrics
import pickle
from typing import Any, List
import random


def GetFiles(path) -> List[str]:
    allFiles = os.walk(path)
    filepath = []
    for (root, _, files) in allFiles:
        for file in sorted(files):
            if (os.path.isfile(os.path.join(root, file)) == True and file != ".DS_Store"):
                filepath.append( os.path.abspath( os.path.join(root, file)))
    
    return filepath


def LoadAudio(path) -> Any:
    filepath = GetFiles(path)
    wave = []
    sample_rate = []
    filename = []
    for index in range(0, len(filepath)):
        sys.stdout.write( '\rLoading...{0}/{1}'.format(index+1, len(filepath)) )
        sys.stdout.flush()
        
        w, r = librosa.load(path=filepath[index])
        wave.append(w)
        sample_rate.append(r)
        filename.append( os.path.basename(filepath[index]) )

    return filename, wave, sample_rate


def GetClasses(filename) -> List[str]:
    result = []
    for classes in filename:
        classes = classes.split('.')[0]
        if classes not in result:
            result.append(classes)

    return result


def GetFeature(wave, classes) -> Any:
    feature = []
    label = []
            
    for index in range( 0, len(wave) ):
        sys.stdout.write( '\rProcessing...{0}/{1}'.format(index+1, len(wave)) )
        sys.stdout.flush()
        temp = wave[index]

        MFCCs = np.average( librosa.feature.mfcc(y=temp) )
        zeroCrossingRate = np.average( librosa.feature.zero_crossing_rate(y=temp) )
        spectralContrast = np.average( librosa.feature.spectral_contrast(y=temp) )
        RMS = np.average(librosa.feature.rms(y=temp))

        feature.append((MFCCs, zeroCrossingRate, spectralContrast, RMS))
        label.append( classes[int(index/50)] )

    return feature, label


def ClassesToNumber(label) -> List[int]:
    result = []
    for index in range(0, len(label)):
        result.append( int(index / 50) )

    return result


def SplitDatasets(data, label, num_classes_data, train_percent) -> Any:
    end = int(num_classes_data * train_percent)

    if random.random() > 0.5:
        train_data = np.array( data[0:end] )
        train_label = np.array( label[0:end] )
        test_data = np.array( data[end:num_classes_data] )
        test_label = np.array( label[end:num_classes_data] )
    else:
        test_data = np.array( data[0:end-30] )
        test_label = np.array( label[0:end-30] )
        train_data = np.array( data[end-30:num_classes_data] )
        train_label = np.array( label[end-30:num_classes_data] )


    for num in range( num_classes_data, len(data), num_classes_data ):
        end += num_classes_data
        if random.random() > 0.5:
            train_data = np.append( train_data, data[num:end], axis=0 )
            train_label = np.append( train_label, label[num:end], axis=0 )
            test_data = np.append( test_data, data[end:num+num_classes_data], axis=0 )
            test_label = np.append( test_label, label[end:num+num_classes_data], axis=0 )
        else:
            train_data = np.append( train_data, data[end-30:num+num_classes_data], axis=0 )
            train_label = np.append( train_label, label[end-30:num+num_classes_data], axis=0 )
            test_data = np.append( test_data, data[num:end-30], axis=0 )
            test_label = np.append( test_label, label[num:end-30], axis=0 )
    
    return train_data, train_label, test_data, test_label


def Serializate(obj, path) -> None:
    outfile = pickle.dumps(obj)
    with open(path, 'wb') as f:
        f.write(outfile)
        f.close()
    
    return


def Deserialize(path) -> sklearn.svm._classes.SVC:
    with open(path, 'rb') as f:
        file = f.read()
        obj = pickle.loads(file)
        f.close()
    
    return obj


def Train(data, label) -> sklearn.svm._classes.SVC:
    model=svm.SVC(probability=True)
    model.fit(data, label)
    Serializate(model, 'svm.model')

    return model


def Test(data, label) -> np.float64:
    model = Deserialize('svm.model')
    predict = model.predict(data)
    # print('Accuracy rate:{0}%'.format( metrics.accuracy_score(label, predict)*100) )

    return metrics.accuracy_score(label, predict)*100


if __name__ == '__main__':
    filename, audio_wave, audio_samplerate = LoadAudio('emotion_analysis_dataset_r')
    Serializate(filename, "filename.data")
    Serializate(audio_wave, "audio_wave.data")
    Serializate(audio_samplerate, "audio_samplerate.data")

    filename = Deserialize("filename.data")
    audio_wave = Deserialize("audio_wave.data")
    audio_samplerate = Deserialize("audio_samplerate.data")

    classes = GetClasses(filename)
    GetFeature(audio_wave, classes)
    feature, label = GetFeature(audio_wave, classes)
    label = ClassesToNumber(label)

    Serializate(feature, 'feature.data')
    Serializate(label, 'label.data')
    
    feature = Deserialize('feature.data')
    label = Deserialize('label.data')

    sum = 0
    for _ in range(1000):
        train_data, train_label, test_data, test_label = SplitDatasets(feature, label, 50, 0.8)
        Train(train_data, train_label)
        sum += Test(test_data, test_label)

    print("\nAverage accuracy rate: {0}%".format(sum/1000))