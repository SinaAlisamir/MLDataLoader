#!/usr/bin/python3
import numpy as np
import os

class MLDataLoader():
    '''
    This class can save and load big data for Machine Learning purposes.\n
    max_samples: maximum number of samples. example: 1000
    feat_shape: define the shape of your features by a tuple like: (30,1) meaning each sample has 30 features
    '''
    def __init__(self, feat_shape, file_name='data', max_samples=1000):
        super(MLDataLoader, self).__init__()
        self.file_name = file_name + '.dat'
        self.max_samples = max_samples
        self.feat_shape = feat_shape
        self.init_shape()
        self.init_reader()
        self.readerIndex = 0

    def init_shape(self):
        state=(self.max_samples,)
        for i in self.feat_shape:
            if i == 1 and self.feat_shape[-1]==i: break
            state=state+(i,)
        self.shape = state

    def init_reader(self):
        try:
            self.reader = np.memmap(self.file_name, dtype='float32', mode='r+', shape=self.shape)
        except:
            self.reader = np.memmap(self.file_name, dtype='float32', mode='w+', shape=self.shape)

    def AddToFile(self, data):
        self.reader[self.readerIndex,:] = data
        self.readerIndex += 1

    def AddToFileWithIndex(self, data, index):
        self.reader[index,:] = data

    def loadFromFile(self, index):
        return self.reader[index]


# rows = 100000
# cols = 4
# filename = 'test.dat'
# shape = (rows,cols)
# data = np.random.rand(rows*cols)
# data.resize(shape)
# # print(data)
#
# dr = MLDataLoader(feat_shape=(cols,1), max_samples=rows)
# dr.AddToFileWithIndex(data[0], 0)
# dr.AddToFileWithIndex(data[1], 1)
# print(dr.loadFromFile(range(2)))

















# fp = np.memmap(filename, dtype='float32', mode='w+', shape=shape)
# for i in range(rows//2):
#     fp[i,:] = data[i]
#
# fp = np.memmap(filename, dtype='float32', mode='r+', shape=shape)
# for i in range(rows//2,rows):
#     fp[i,:] = data[i]
#
# fpo = np.memmap(filename, dtype='float32', mode='r')
# fpo.resize(shape)
# print(fpo[rows//2+2000])
#
# print((2,(5,4)[1]))
# a=(5,4)
# print(len(a))
# state=(12,)
# for i in a:
#     state=state+(i,)
# print(state)



#
