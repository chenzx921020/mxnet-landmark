#!/usr/bin/env python
# coding=utf-8
import sys,os
import mxnet as mx      
import cv2
import numpy as np      
from collections import namedtuple
import mixup



if __name__ == '__main__':   
    Batch = namedtuple('Batch', ['data'])
    sym,arg_params,aux_params = mx.model.load_checkpoint('tmp/model',600)
    print sym.get_internals()
    mod = mx.mod.Module(symbol=sym,context=mx.cpu(),data_names=['data'],label_names=[])
    mod.bind(for_training=False,data_shapes=[('data',(1,1,72,72))],label_shapes=mod._label_shapes)
    mod.set_params(arg_params,aux_params)
    img_url = sys.argv[1] 
    
    img = cv2.cvtColor(cv2.imread(img_url), cv2.COLOR_BGR2GRAY)
    img = (img-128)*0.01
    #print img.shape
    #img = cv2.resize(img, (72, 72))
    
    #img = np.swapaxes(img, 0, 1)
    #img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    img = img[np.newaxis, :]
    #model forward compute
    mod.forward(Batch([mx.nd.array(img)]))
    
    #feature extraction
    '''
    all_layers = sym.get_internals()
    fe_sym = all_layers[40]
    print fe_sym
    fe_mod = mx.mod.Module(symbol=fe_sym,context=mx.cpu(),label_names=[])
    fe_mod.bind(for_training=False, data_shapes=[('data',(1,1,72,72))])
    fe_mod.set_params(arg_params,aux_params)
    fe_mod.forward(Batch([mx.nd.array(img)]))
    features = fe_mod.get_outputs()[0].asnumpy()
    sum_fe = sum(sum(features))/features.shape[1]*100+128
    print sum_fe
    print sum_fe.shape
    sum_fe = np.array(sum_fe).astype("uint8")
    cv2.imshow('feature',sum_fe)
    cv2.waitKey(0)

    '''
    #result get
    prob = mod.get_outputs()[0].asnumpy()    
    prob = np.squeeze(prob)
    print prob
    print prob.shape
    #print result on image
    new_img=cv2.imread(img_url)
    #new_img=cv2.resize(new_img,(72,72))
    for i in range(0,21):
        cv2.circle(new_img,(int(prob[2*i]),int(prob[2*i+1])),2,(0,255,0))
    cv2.imshow('test',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
