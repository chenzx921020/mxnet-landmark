import os, sys
import argparse
import find_mxnet
import mxnet as mx
import qnn_absorb_bn as qnn
import alphacnn as alpha


def ConvFactory(name, data, num_filter, kernel, stride=1, pad=0, shifts, use_activation=True, vector_quanti=False):
    conv = qnn.QuantiConv(data=data, num_filter=num_filter,
                                 kernel=(kernel, kernel),
                                 stride=(stride, stride),
                                 pad=(pad, pad),
                                 shifts=shifts,
                                 use_activation=use_activation,
                                 vector_quanti=vector_quanti,
                                 name='conv_{}'.format(name))
    return conv


def get_symbol(is_train=True, vector_quanti=False, inference=False, lmk_count=21, pose_count=3):
    base_number = 128
    shifts = {'w':7, 'b':7, 'o':4}
    data = mx.symbol.Variable("data")
    data = alpha.QuantiInput(data=data, name="quantiinput0", fixed_output_shift=7)
    label = mx.symbol.Variable("softmax_label")

    conv1 = ConvFactory('conv1', data, 16, 3, 2, 0, shifts, True, vector_quanti)
    conv1_2 = ConvFactory('conv1_2', conv1, 16, 3, 2, 0, True, vector_quanti)
    conv2 = ConvFactory('conv2', conv1_2, 32, 3, 2, 0, True, vector_quanti)
    conv3 = ConvFactory('conv3', conv2, 32, 3, 1, 1, True, vector_quanti)
    conv4 = ConvFactory('conv4', conv3, 32, 3, 1, 0, True, vector_quanti)
    conv5 = ConvFactory('conv5', conv4, 16, 3, 2, 1, True, vector_quanti)
    conv6_1 = ConvFactory('conv6_1', conv5, 48, 3, 1, 1, True, vector_quanti)
    conv6_2 = ConvFactory('conv6_2', conv6_1, 48, 3, 1, 1, True, vector_quanti)
    conv6_2 = mx.symbol.QuantiFlatten(data=conv6_2)

    # pose
    fc6 = qnn.QuantiFC(shifts=shifts, vector_quanti=vector_quanti, data=conv6_2, num_hidden=64)
    act6 = mx.symbol.QuantiActivation(data=fc6, act_type='relu')
    fc7 = qnn.QuantiFC(shifts=shifts, vector_quanti=vector_quanti, data=act6, num_hidden=pose_count)

    if is_train:
        loss_pose = mx.symbol.LinearRegressionOutput(data=fc7, label=label, use_ignore=True, grad_scale=1)
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
