# -*- coding: utf-8 -*-
import os
import csv
import math
import numpy
import numpy as np
import six.moves.cPickle as pickle

#---Chainer---
import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import cupy

#%%
MODE_TRAIN = True
MODE_TEST = False
MODE_PREDICT = False
#%%

class NetClass:
    xp = numpy
    mode = 0
    use_gpu = False
    use_dev = 0
    optimizer = 0
    loss = 0
    
    input_shape = 0
    input_w = 0
    input_h = 0
    input_c = 0
    input_num = 0
    hidden_num = 0
    output_num = 0
    unit_num = 0   
    out_w = 0
    out_h = 0
    k_w = 0
    k_h = 0
    
    tmp_x = 0    
    tmp_t = 0
    tmp_E = 0
    tmp_y = 0
    X = 0
    T = 0
    Y = 0
    E = 0
    
    conv_h = 0
    line1_h = 0
    lstm_h = 0
    line2_h = 0
    reshape_h = 0
    
    model = 0
    def __init__(self,in_n,hid_n,out_n,in_data):
        self.mode = 0
        self.input_num = in_n
        self.hidden_num = hid_n
        self.output_num = out_n
        self.input_shape = in_data
        n,self.input_c,self.input_w,self.input_h = in_data.shape
        print in_data.shape
        
        self.k_h = 4
        self.k_w = 4       
        self.out_h = (self.input_h-2*(self.k_h/2)+1)
        self.out_w = (self.input_w-2*(self.k_w/2)+1)
        self.unit_num = hid_n*self.out_w*self.out_h
        print self.unit_num
        self.defineModel()
    
    def setCudaPath(self):
        path = os.environ["PATH"]
        cuda_path = ':/usr/local/cuda/bin'
        if path.find(cuda_path) == -1:
            os.environ['PATH'] += cuda_path  #CUDAに通す
            
    def setup(self,gpu=False,dev=0):        
        if gpu == True:
            self.setCudaPath()
            self.xp = cupy
            self.use_gpu = True
            self.use_dev = dev
            cuda.get_device(dev).use()
            self.model.to_gpu()
        self.E = self.convData(self.input_shape,self.mode)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())
        
    #---model function---    
    def defineModel(self):
        lstm_unit = self.unit_num
        self.model = chainer.FunctionSet(
            conv=F.Convolution2D(self.input_num, self.hidden_num, (self.k_w,self.k_h),pad=0),
            line1=F.Linear(self.unit_num,lstm_unit),
            lstm=L.LSTM(lstm_unit,lstm_unit),
            line2=F.Linear(lstm_unit,self.unit_num),
            deconv=L.Deconvolution2D(self.hidden_num,self.output_num,(self.k_w,self.k_h),pad=0)
            )
          
    def resetLSTM(self):
        self.model.lstm.reset_state()
        self.E = self.convData(self.input_shape,self.mode)
        
    #---forward function---        
    def forward(self,A,E):
        self.conv_h = F.relu(self.model.conv(E))
        self.line1_h = F.relu(self.model.line1(self.conv_h))
        self.lstm_h = self.model.lstm(self.line1_h)
        self.line2_h = F.relu(self.model.line2(self.lstm_h))
        self.reshape_h = F.reshape(self.line2_h,(1,self.hidden_num,self.out_w,self.out_h))
        self.tmp_y = F.relu(self.model.deconv(self.reshape_h))
        
        e0 = F.relu(A-self.tmp_y)
        e1 = F.relu(self.tmp_y-A)
        self.tmp_E = F.concat([e0,e1],1)
        
        return self.tmp_y,self.tmp_E
        
    #---data function---
    def convData(self,x,m=True):
        if self.use_gpu == True:
            x = cuda.to_gpu(x)
        return chainer.Variable(x)
    
    def toCPU(self,x):
        x = cuda.to_cpu(x)
        return x
    
    #---main function---
    def train(self,x,t):
        self.mode = MODE_TRAIN
        self.T = self.convData(t,self.mode)
        self.X = self.convData(x,self.mode)
        self.optimizer.zero_grads()       # 勾配をゼロ初期化
        self.Y,self.E = self.forward(self.X,self.E)
        self.calcLoss(self.E,self.T)
        
    def predict(self,x):
        self.mode = MODE_TEST
        self.X = self.convData(x,self.mode)
        self.Y,self.E = self.forward(self.X,self.E)
        
    def calcLoss(self,y,t):
        if self.mode == MODE_TRAIN:
            self.loss = F.mean_squared_error(y,t)
            self.loss.backward()  # 誤差逆伝播
            self.optimizer.update()     # 最適化 
        else :
            self.loss = F.mean_squared_error(y,t)
        
    def calcAccuracy(self,y,t):
        return F.accuracy(y,t)
        
    #---File function---
    def saveModel(self,file_name):
        if self.use_gpu == True:        
            self.model.to_cpu()
        pickle.dump(self.model,open(file_name+'.model','wd'))
        if self.use_gpu == True:
            self.model.to_gpu()

    def loadModel(self,file_name):
        self.model =  pickle.load(open(file_name+'.model','rb'))
        
#%%
