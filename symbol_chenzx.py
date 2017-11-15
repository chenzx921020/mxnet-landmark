import os, sys
import argparse
import find_mxnet
import mxnet as mx


def ConvFactory(name, data, num_filter, kernel, stride=1, pad=0, use_act=True):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter,
                                 kernel=(kernel, kernel),
                                 stride=(stride, stride),
                                 pad=(pad, pad),
                                 name='conv_{}'.format(name))
    # bn = mx.symbol.BatchNorm(data=conv, name='bn_{}'.format(name))
    if use_act == True:
        act = mx.sym.Activation(data=conv, act_type='relu',
                                   name='act_{}'.format(name))
        return act
    else:
        return conv


def get_symbol(is_train=True, lmk_count=21, pose_count=3):
    data = mx.sym.Variable("data")
    label = mx.sym.Variable("softmax_label")
    conv1 = ConvFactory('conv1', data, 16, 3, 2, 0)
    conv1_2 = ConvFactory('conv1_2', conv1, 32, 3, 2, 0)
    conv2 = ConvFactory('conv2', conv1_2, 32, 3, 1, 0)
    conv3 = ConvFactory('conv3', conv2, 48, 3, 2, 1)
    conv4 = ConvFactory('conv4', conv3, 48, 3, 1, 0)
    conv5 = ConvFactory('conv5', conv4, 64, 3, 1, 1)
    conv6_1 = ConvFactory('conv6_1', conv5, 64, 3, 1, 1)
    
    conv6_2 = ConvFactory('conv6_2', conv6_1, lmk_count*2, 3, 1, 0,False)
    #conv7 = ConvFactory('conv7', conv6_2, 16, 3, 1, 1)
    #conv8 = ConvFactory('conv8', conv7, 8, 3, 1, 1)
    #conv9 = ConvFactory('conv9', conv8, 3, 3, 1, 0)
    #con10 = ConvFactory('con10', conv9, 3, 1, 1, 0)
    conv10 = mx.symbol.Pooling(data=conv6_2, global_pool=True,pool_type='avg',kernel=(1,1))
    conv10_2 = mx.symbol.Flatten(data=conv10)


    if is_train:
        offset = 0
        
        label_lmk = mx.symbol.slice_axis(data=label[0], axis=1, begin=offset, end=offset+lmk_count*2)
        label_lmk = mx.symbol.Flatten(data=label_lmk)
        loss_lmk = mx.sym.LinearRegressionOutput(data=conv10_2, label=label_lmk, grad_scale=1)
        offset=offset+lmk_count*2
        return loss_lmk
        #offset=offset+lmk_count*2+4

    return conv10_2


if __name__ == '__main__':
    data_names = ['data']
    label_names = ['label', 'mask']
    data = mx.symbol.Variable(name=data_names[0])
    labels = [mx.symbol.Variable(name=name) for name in label_names]

    # can not be together
    is_train = False
    symbol = get_symbol(is_train=is_train)
    print symbol.list_arguments()
    symbol.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'symbol_%s.json' % ('train' if is_train else 'test')))

    #a = mx.viz.plot_network(symbol, shape={"data":(1, 1, 72, 72)}, node_attrs={"shape":'rect',"fixedsize":'false'})
    #a.render()
