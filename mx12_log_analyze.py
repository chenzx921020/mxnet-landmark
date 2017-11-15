# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:18:03 2017

@author: root
"""

import os,sys
import numpy as np
import pylab

log_path = sys.argv[1]
save_path = sys.argv[2]
lines = open(log_path,'r').readlines()
train_loss = []
train_list = []
for i in range(0,len(lines)):
    tmp = lines[i].split(' ')
    
    if ('Epoch' ==tmp[2][:5]) and ('Train-LmkMetric_0'==tmp[3][:17]):

        train_loss.append(float(tmp[-1].split('=')[-1]))
        #train_list.append(i)
del train_loss[0]
pylab.plot(train_loss)
pylab.xlabel('epoch')
pylab.ylabel('train_loss')
#pylab.xlim([0, 1])
#pylab.ylim([0, 1])
#pylab.text(rec[100],pre[100])
pylab.xticks(np.arange(0.,500,50), fontsize = 8)
pylab.yticks(np.arange(0.,1,0.1), fontsize = 8)
pylab.title('linear_regression_loss')
pylab.grid()
#pic_name = sys.argv[3]
pylab.savefig(save_path + 'mx12_train_loss.jpg')
pylab.close('all')