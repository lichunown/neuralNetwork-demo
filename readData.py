# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:27:24 2017

@author: lichunyang
"""
import struct
import numpy as np
import matplotlib.pyplot as plt

filename = 'train-images.idx3-ubyte'

def showImage(img2):
    img = img2.copy()
    img.shape=(28,28)
    plt.imshow(img)
    plt.show()

def int32(data):
    return struct.unpack('i',data)
"""
def readImages(filename):
    f = open(filename,'rb')
    MSB = int32(f.read(4)[::-1])
    numofitems = int32(f.read(4)[::-1])[0]
    rows = int32(f.read(4)[::-1])[0]
    columns = int32(f.read(4)[::-1])[0]
    data = np.zeros([numofitems,rows,columns],'uint8')
    temp = f.read(1)
    tn = 0
    tr = 0
    tc = 0
    while tn < numofitems:
        data[tn,tr,tc] = np.uint8(struct.unpack('B',temp))
        temp = f.read(1)
        if tc == columns-1:
            tc = 0
            tr += 1
        else:
            tc += 1
            continue
        if tr == rows-1:
            tr = 0
            tn += 1
        else:
            tr += 1
            continue
    return data
"""


def readImages(filename):
    f = open(filename,'rb')
    MSB = int32(f.read(4)[::-1])
    numofitems = int32(f.read(4)[::-1])[0]
    rows = int32(f.read(4)[::-1])[0]
    columns = int32(f.read(4)[::-1])[0]
    data = np.zeros([numofitems,rows,columns,1],'float32')
    for num in range(numofitems):
        temp = f.read(rows*columns)
        img = np.array(struct.unpack('B'*rows*columns,temp),'int8')
        img.shape=(28,28,1)
        data[num,:,:] = img
    return data/255.0
"""
def readImages(filename):
    f = open(filename,'rb')
    MSB = int32(f.read(4)[::-1])
    numofitems = int32(f.read(4)[::-1])[0]
    rows = int32(f.read(4)[::-1])[0]
    columns = int32(f.read(4)[::-1])[0]
    data = np.zeros([numofitems,rows,columns],'uint8')
    temp = f.read(rows*columns*numofitems)
    img = np.array(struct.unpack('B'*rows*columns*numofitems,temp),'int8')
    img.shape=(numofitems,28,28)
    return img
"""

'''
def readImages2(filename):
    f = open(filename,'rb')
    MSB = int32(f.read(4)[::-1])
    numofitems = int32(f.read(4)[::-1])[0]
    rows = int32(f.read(4)[::-1])[0]
    columns = int32(f.read(4)[::-1])[0]
    data = np.zeros([numofitems,rows*columns],'uint8')
    temp = f.read(1)
    tn = 0
    tr = 0
    while tn < numofitems:
        data[tn,tr] = np.uint8(struct.unpack('B',temp))
        temp = f.read(1)
        if tr == rows*columns-1:
            tr = 0
            tn += 1
        else:
            tr += 1
            continue
    return data/255.0
 '''
def readImages2(filename):
    f = open(filename,'rb')
    MSB = int32(f.read(4)[::-1])
    numofitems = int32(f.read(4)[::-1])[0]
    rows = int32(f.read(4)[::-1])[0]
    columns = int32(f.read(4)[::-1])[0]
    data = np.zeros([numofitems,rows*columns],'uint8')
    for i in range(numofitems):
        temp = f.read(rows*columns)
        data[i,:] = np.array(struct.unpack('B'*rows*columns,temp),'int8')
    return data/255.0

def readLabels(filename):
    f = open(filename,'rb')
    MSB = struct.unpack('i',f.read(4))
    numofitems = struct.unpack('i',f.read(4)[::-1])
    datanum = np.zeros(numofitems,'uint8')
    temp = f.read(1)
    i = 0
    while temp:
        datanum[i] = np.uint8(struct.unpack('B',temp))
        i += 1
        temp = f.read(1)
    return datanum

def readLabels2(filename):
    f = open(filename,'rb')
    MSB = struct.unpack('i',f.read(4))
    numofitems = int32(f.read(4)[::-1])[0]
    temp = f.read(numofitems)
    datanum = np.array([struct.unpack('B'*numofitems,temp)],'int8').T
    return datanum
    

def oneHot(label):
    result = np.zeros([label.shape[0],10])
    for i in range(label.shape[0]):
        result[i,label[i]]=1
    return result

def getRandomData(data1,data2,num):
    H = data1.shape[0]
    outH = np.random.permutation(H)
    outH = outH[0:num]
    return (data1[outH,:],data2[outH,:])