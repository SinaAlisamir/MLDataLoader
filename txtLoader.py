#!/usr/bin/python3
from MLDataLoader import MLDataLoader
import glob
import os
import sys
import python_speech_features
import soundfile as sf
import numpy as np
from funcs import toOneHot, seqLenFromVec

class txtLoader(MLDataLoader):
    def __init__(self, files, feat_shape, file_name='txt_data', max_samples=1000):
        super(txtLoader, self).__init__(feat_shape, file_name=file_name, max_samples=max_samples)
        self.files = files

    lexicon = ['iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey',
                'ay', 'oy', 'aw', 'ow', 'l', 'r', 'y', 'w', 'er', 'm', 'n',
                'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx', 'g', 'p', 't', 'k',
                'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil']

    def get_txts(self):
        for (i,txtFile) in enumerate(self.files):
            line = open(txtFile).read()
            labels = [self.lexicon.index(label) for label in line.split()]
            diff_len = self.feat_shape[0]-len(labels)
            for _ in range(diff_len):
                labels.append(-1)
            self.AddToFileWithIndex(np.array(labels), i)

path = 'timit_data/test_core'
files = glob.glob(os.path.join(path,"*.txt"), recursive=True)
myTxts = txtLoader(files=files, feat_shape=(100,1), max_samples=200)
# print(os.path.exists('txt_data.dat'))
myTxts.get_txts()
print(toOneHot(myTxts.reader[0], one_hot_size=40))
print(myTxts.reader[0])
new_vec, seq_len = seqLenFromVec(myTxts.reader[0])
print(new_vec, seq_len)
