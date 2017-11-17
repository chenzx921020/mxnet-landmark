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
    sym,arg_params,aux_params = mx.model.load_checkpoint('mixup3/model',300)
    print sym.get_internals()
    mod = mx.mod.Module(symbol=sym,context=mx.cpu(),data_names=['data'],label_names=[])
    mod.bind(for_training=False,data_shapes=[('data',(1,1,72,72))],label_shapes=mod._label_shapes)
    mod.set_params(arg_params,aux_params)
    img_url = sys.argv[1] 
    
    img = cv2.cvtColor(cv2.imread(img_url), cv2.COLOR_BGR2GRAY)
    #print img.shape
    img = cv2.resize(img, (72, 72))
    #img = np.swapaxes(img, 0, 2)
    #img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    img = img[np.newaxis, :]
    #model forward compute
    mod.forward(Batch([mx.nd.array(img)]))
    
    #feature extraction
    #all_layers = sym.get_internals()
    #all_layers.list_outputs()[-10:]    
    #print all_layers
    
    #fe_sym = all_layers[:12]
    
    #print fe_sym
    #fe_mod = mx.mod.Module(symbol=fe_sym, context=mx.cpu(), data_names=['data'], label_names=None)
    #fe_mod.bind(for_training=False, data_shapes=[('data', (1,1,72,72))],label_shapes=fe_mod._label_shapes)
    #fe_mod.set_params(arg_params, aux_params)
    #fe_mod.forward(Batch([mx.nd.array(img)]))
    #features = fe_mod.get_outputs()[0].asnumpy()
    #feature_map = features[0][0].astype("uint8")
    #print feature_map.shape
    
    #cv2.imshow('feature',feature_map)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print features[0][0]
    
    
    #result get
    prob = mod.get_outputs()[0].asnumpy()    
    prob = np.squeeze(prob)
    print prob 
    #print result on image
    new_img=cv2.imread(img_url)
    for i in range(0,21):
        cv2.circle(new_img,(int(prob[2*i]),int(prob[2*i+1])),2,(0,255,0))
    cv2.imshow('test',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
