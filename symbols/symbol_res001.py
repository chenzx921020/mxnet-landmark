import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import argparse
import find_mxnet
import mxnet as mx

### ResNet
def Res_S(data, ch_data, ch_3x3x2_1, ch_3x3x2_2):
    act_0 = mx.symbol.Activation(data=data, act_type='relu')

    conv_1 = mx.symbol.Convolution(data=act_0, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3x2_1)
    act_1  = mx.symbol.Activation(data=conv_1, act_type="relu")

    conv_2 = mx.symbol.Convolution(data=act_1, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3x2_2)

    esum = mx.symbol.ElementWiseSum(data, conv_2)
    return esum

def Res_D(data, ch_data, ch_3x3x2_1, ch_3x3x2_2):
    act_0 = mx.symbol.Activation(data=data, act_type='relu')

    conv_1 = mx.symbol.Convolution(data=act_0, kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=ch_3x3x2_1)
    act_1  = mx.symbol.Activation(data=conv_1, act_type="relu")

    conv_2 = mx.symbol.Convolution(data=act_1, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3x2_2)

    proj = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=ch_3x3x2_2)

    esum = mx.symbol.ElementWiseSum(proj, conv_2)
    return esum

def Pool_D(data, ch_data, ch_3x3x2_1, ch_3x3x2_2):
    act_0 = mx.symbol.Activation(data=data, act_type='relu')

    conv_a = mx.symbol.Convolution(data=act_0, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=ch_3x3x2_1)
    act_a = mx.symbol.Activation(data=conv_a, act_type='relu')

    pool_a = mx.symbol.Pooling(data=act_a, kernel=(3, 3), pad=(1, 1), stride=(2, 2), pool_type='max')

    conv_b = mx.symbol.Convolution(data=pool_a, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=ch_3x3x2_2)

    proj = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=ch_3x3x2_2)

    esum = mx.symbol.ElementWiseSum(proj, conv_b)
    return conv_b

def get_symbol(is_train=True, lmk_count=21, pose_count=3):
    data = mx.symbol.Variable("data")
    labels = mx.symbol.Variable("softmax_label")

    conv0 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(0, 0), stride=(2, 2), num_filter=32)
    res1a = Res_S(conv0, 32, 32, 32)
    res1b = Res_D(res1a, 32, 48, 48)

    res2a = Res_S(res1b, 48, 48, 48)
    res2b = Res_D(res2a, 48, 64, 64)

    res3a = Res_S(res2b, 64, 64, 64)
    res3b = Pool_D(res3a, 64, 80, 80)

    res4a = Res_S(res3b, 80, 80, 80)
    res4b = Pool_D(res4a, 80, 96, 96)
    act4 = mx.symbol.Activation(data=res4b, act_type='relu')

    # stage concat
    flatten = mx.symbol.Flatten(data=act4)
    # fc_lmk = mx.symbol.FullyConnected(data=flatten, num_hidden=lmk_count*2)
    # fc_attr = mx.symbol.FullyConnected(data=flatten, num_hidden=lmk_count)
    fc_pose = mx.symbol.FullyConnected(data=flatten, num_hidden=pose_count)

    if is_train:
        # offset = 0
        # label_lmk = mx.symbol.slice_axis(data=labels, axis=1, begin=0, end=lmk_count*2)
        # label_lmk = mx.symbol.Flatten(data=label_lmk)
        # loss_lmk = mx.symbol.LinearRegressionOutput(data=fc_lmk, label=label_lmk, grad_scale=1)
        # offset += lmk_count * 2

        # label_attr = mx.symbol.slice_axis(data=labels, axis=1, begin=offset, end=offset+lmk_count)
        # label_attr = mx.symbol.Flatten(data=label_attr)
        # loss_attr = mx.symbol.LinearRegressionOutput(data=fc_attr, label=label_attr, grad_scale=1)
        # offset += lmk_count

        label_pose = labels
        # label_pose = mx.symbol.slice_axis(data=labels, axis=1, begin=63, end=66)
        label_pose = mx.symbol.Flatten(data=label_pose)
        loss_pose = mx.symbol.LinearRegressionOutput(data=fc_pose, label=label_pose, grad_scale=1)
        # offset += pose_count
        # return mx.symbol.Group([loss_lmk])
        return mx.symbol.Group([loss_pose])
        # return mx.symbol.Group([loss_lmk, loss_attr, loss_pose])
    return mx.symbol.Group([fc_pose])
    # return mx.symbol.Group([fc_lmk, fc_attr, fc_pose])

if __name__ == '__main__':
    data_names = ['data']
    label_names = ['softmax_label']
    data = mx.symbol.Variable(name=data_names[0])
    labels = [mx.symbol.Variable(name=name) for name in label_names]

    # can not be together
    is_train = False
    symbol = get_symbol(is_train=is_train, lmk_count=21)
    symbol.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'symbol_%s.json' % ('train' if is_train else 'test')))
