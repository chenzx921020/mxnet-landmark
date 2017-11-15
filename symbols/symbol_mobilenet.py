import mxnet as mx

def _sep_conv_mod(data, suffix, filter_list,num_filter, downsample=False):
    _stride = 2 if downsample else 1
    conv1_dw =  mx.symbol.Convolution(name='conv1_{}_dw'.format(suffix), data=data, 
            num_filter=filter_list[-1], pad=(1, 1), kernel=(3,3), 
            stride=(_stride,_stride), no_bias=True, num_group=filter_list[-1])
    conv1_dw_bn = mx.symbol.BatchNorm(name='conv1_{}_dw_bn'.format(suffix),
            data=conv1_dw,fix_gamma=False, eps=0.00010)
    conv1_dw_scale = conv1_dw_bn
    relu1_dw = mx.symbol.Activation(name='relu1_{}_dw'.format(suffix),
            data=conv1_dw_scale,act_type='relu')

    conv1_sep = mx.symbol.Convolution(name='conv1_{}_sep'.format(suffix), 
            data=relu1_dw , num_filter=num_filter, pad=(0, 0), 
            kernel=(1,1), stride=(1,1), no_bias=True)
    conv1_sep_bn = mx.symbol.BatchNorm(name='conv1_{}_sep_bn'.format(suffix),
            data=conv1_sep, fix_gamma=False, eps=0.00010)
    conv1_sep_scale = conv1_sep_bn
    relu_sep = mx.symbol.Activation(name='relu1_{}_sep'.format(suffix),
            data=conv1_sep_scale,act_type='relu')
    
    filter_list.append(num_filter)
    return relu_sep
def get_symbol(num_classes):
    data = mx.symbol.Variable("data")
    label = mx.symbol.Variable("softmax_label")
    conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=8, 
            pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
    conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', 
            data=conv1 , fix_gamma=False, eps=0.000100)
    conv1_scale = conv1_bn
    net = mx.symbol.Activation(name='relu1', 
            data=conv1_scale , act_type='relu')
    filter_list = [8]
    net = _sep_conv_mod(net, 1, filter_list, 16)
    net = _sep_conv_mod(net, 2, filter_list, 32, True)
    net = _sep_conv_mod(net, 3, filter_list, 32)     
    net = _sep_conv_mod(net, 4, filter_list, 32, True)
#    net = _sep_conv_mod(net, 5, 256, 256)
#    net = _sep_conv_mod(net, 6, 256, 512, True)

    net = _sep_conv_mod(net, 7, filter_list, 64)
    
    pool6 = mx.symbol.Pooling(name='pool6', data=net , 
            pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
    fc7 = mx.symbol.Convolution(name='fc7', data=pool6 , 
            num_filter=num_classes, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
    flatten = mx.symbol.Flatten(data=fc7, name='flatten')
    output = mx.symbol.LinearRegressionOutput(data=flatten,
            label = label,name='linearregression')
    return output

if __name__ == '__main__':
    symbol = get_symbol(3)
    a = mx.viz.plot_network(symbol, shape={"data":(1, 1, 72, 72)}, node_attrs={"shape":'rect',"fixedsize":'false'})
    a.render()


