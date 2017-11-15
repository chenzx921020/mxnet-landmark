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
    conv2 = ConvFactory('conv2', conv1_2, 32, 3, 2, 0)
    conv3 = ConvFactory('conv3', conv2, 32, 3, 1, 1)
    conv4 = ConvFactory('conv4', conv3, 32, 3, 1, 0)
    conv5 = ConvFactory('conv5', conv4, 64, 3, 2, 1)
    conv6_1 = ConvFactory('conv6_1', conv5, 64, 3, 1, 1)
    conv6_2 = ConvFactory('conv6_2', conv6_1, 64, 3, 1, 1)
    # conv6_2 = mx.symbol.Flatten(data=conv6_2)

    # pose
    # fc6 = mx.symbol.FullyConnected(data=conv6_2, num_hidden=64)
    # act6 = mx.symbol.Activation(data=fc6, act_type='relu')
    # fc7 = mx.symbol.FullyConnected(data=act6, num_hidden=pose_count)
    fc6 = ConvFactory('fc6', conv6_2, 64, 3, 1, 0)
    fc7 = ConvFactory('fc7', fc6, pose_count, 1, 1, 0, False)
    fc7 = mx.sym.Flatten(data=fc7)

    if is_train:
        loss_pose = mx.sym.LinearRegressionOutput(data=fc7, label=label, grad_scale=1)
        return loss_pose

    return fc7


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

    a = mx.viz.plot_network(symbol, shape={"data":(1, 1, 72, 72)}, node_attrs={"shape":'rect',"fixedsize":'false'})
    a.render()
