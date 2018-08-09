#!/usr/bin/python3
from MLDataLoader import MLDataLoader
import glob
import os
import sys
import python_speech_features
import soundfile as sf
import numpy as np
from funcs import seqLenFromVecs

class wavLoader(MLDataLoader):
    def __init__(self, files, feat_shape, file_name='wav_data', max_samples=1000):
        super(wavLoader, self).__init__(feat_shape, file_name=file_name, max_samples=max_samples)
        self.files = files

    def changeExtention(self, fromExt, toExt):
        ext_list = []
        for path in self.files:
            new_path = path[0:len(path)-len(fromExt)]+toExt
            ext_list.append(new_path)
        self.files = ext_list

    def normalize(self, feats):
        result = feats
        if feats.std(axis=0) != 0: result = (feats - feats.mean(axis=0)) / feats.std(axis=0)
        return result

    def makeSegments(self, win_length=0.025, win_shift=0.01, normed=False):
        self.win_length = win_length
        self.win_shift = win_shift
        sample_length = int(win_length*16000)
        sample_shift = int(win_shift*16000)
        for (i,wave) in enumerate(self.files):
            theFileData = []
            wav_read = sf.read(wave)[0]
            wavs = python_speech_features.sigproc.framesig(wav_read, sample_length, sample_shift)
            for wav in wavs:
                theWav = wav
                if normed: theWav = self.normalize(wav)
                theFileData.append(theWav)
            diff_len = self.feat_shape[0]-len(theFileData)
            for _ in range(diff_len):
                theFileData.append(np.zeros(sample_length))
            array = np.array(theFileData)
            self.AddToFileWithIndex(array, i)

    def makeMFCCs(self, win_length=0.025, win_shift=0.01, context_size=4):
        self.win_length = win_length
        self.win_shift = win_shift
        for (i,wave) in enumerate(self.files):
            theFileData = []
            wav_read = sf.read(wave)[0]
            mfccs = python_speech_features.base.mfcc(wav_read, winlen=self.win_length, winstep=self.win_shift, numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=np.hamming)
            mfcc_norms = [self.normalize(mfcc) for mfcc in mfccs]
            delta1 = python_speech_features.base.delta(mfccs, context_size)
            delta1_norms = [self.normalize(delta) for delta in delta1]
            delta2 = python_speech_features.base.delta(delta1, context_size)
            delta2_norms = [self.normalize(delta) for delta in delta2]
            for (j,_) in enumerate(mfccs):
                feats = np.append(mfcc_norms[j], np.append(delta1_norms[j],delta2_norms[j]))
                theFileData.append(feats)
            diff_len = self.feat_shape[0]-len(theFileData)
            length = len(theFileData[0])
            for _ in range(diff_len):
                theFileData.append(np.zeros(length))
            array = np.array(theFileData)
            self.AddToFileWithIndex(array, i)

path = 'timit_data/test_core'
files = glob.glob(os.path.join(path,"*.txt"), recursive=True)
myWaves = wavLoader(files=files, feat_shape=(5000,400), max_samples=200)
myWaves.changeExtention(fromExt='txt', toExt='wav')
# print(os.path.exists('wav_data.dat'))
# myWaves.makeSegments(win_length=0.025, win_shift=0.01)
print(len(myWaves.reader[0]))
new_vecs, seq_len = seqLenFromVecs(myWaves.reader[0])
print(new_vecs, seq_len)
