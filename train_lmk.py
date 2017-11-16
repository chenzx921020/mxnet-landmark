import sys, os
import find_mxnet
import logging
import mxnet as mx
import numpy as np
import symbol_chenzx as net
from lmk_pose_metric import LmkMetric2
#from lmk_pose_metric import PoseMetric

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


lmk_count = 21
ocular_pt_ind1 = 7
ocular_pt_ind2 = 10


root_dir = "/home/users/zhixuan.chen/data/"
batch_size = 64
input_channel = 1
input_size = 72
input_shape = (input_channel, input_size, input_size)


def get_test_iterator():
    test_dataiter = mx.io.ImageRecordIter(
        # path_imglist = root_dir + "/annotations/face_rect/umd_batch3.pose.txt.nnvm",
        # path_imgrec = root_dir + "/img_rec/face_rect/umd.batch3.pose.gray.rec",
        # label_width = pose_count,
        path_imglist = "/home/users/zhixuan.chen/data/val.lst",
        path_imgrec = "/home/users/zhixuan.chen/data/val.rec",
        label_width = lmk_count*2,
        data_shape  = input_shape,
        batch_size  = batch_size,
        mean_r = 128,
        #mean_g = 128,
        #mean_b = 128,
        scale = 0.01,
    )
    return test_dataiter


def get_data_iterator():
    train_dataiter = mx.io.ImageRecordIter(
        # path_imglist = root_dir + "/annotations/face_rect/umd_batch1.pose.txt.nnvm",
        # path_imgrec = root_dir + "/img_rec/face_rect/umd.batch1.pose.gray.rec",
        # label_width = pose_count,
        path_imglist = "/home/users/zhixuan.chen/data/train.lst",
        path_imgrec = "/home/users/zhixuan.chen/data/train.rec",
        label_width = lmk_count*2,
        data_shape  = input_shape,
        #shuffle     = True,
        batch_size  = batch_size,
        # rand_crop   = True,
        # max_rotate_angle = 10,
        # max_aspect_ratio = 0.02,
        # max_shear_ratio = 0.1,
        # max_crop_size = 72,
        # min_crop_size = 65,
        # max_random_scale = 1.1,
        # min_random_scale = 0.9,
        # random_h = 100,
        # random_s = 100,
        # random_l = 100,
        # rand_mirror = True,
        mean_r = 128,
        #mean_g = 128,
        #mean_b = 128,
        scale = 0.01,)
    test_dataiter = get_test_iterator()
    return (train_dataiter, test_dataiter)


save_dir = "./tmp"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
model_prefix = save_dir + "/model"


def fit(sym, train, val, batch_size, num_gpus):
    load_epoch = -1
    train_epoch = 400
    arg_params = None
    aux_params = None
    if load_epoch != -1:
        print 'continue training with epoch %d' % (load_epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, load_epoch)
        train_epoch += load_epoch
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=sym, context=devs)
    metric = LmkMetric2( 0,lmk_count, ocular_pt_ind1, ocular_pt_ind2)
    # metric = PoseMetric(pose_count)
    mod.fit(train, val,
            begin_epoch        = 0 if load_epoch == -1 else load_epoch,
            num_epoch          = train_epoch,
            allow_missing      = True,
            batch_end_callback = mx.callback.Speedometer(batch_size, 64),
            kvstore            ='device',
            optimizer          ='adam',
            optimizer_params   = {'learning_rate':0.1,'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=500,factor=0.4)},
            initializer        = mx.init.Xavier(rnd_type='uniform', factor_type="avg", magnitude=2.34),
            epoch_end_callback = mx.callback.do_checkpoint(model_prefix),
            eval_metric        = metric,
            arg_params         = arg_params,
            aux_params         = aux_params,)
    return mod.score(val, metric)


if __name__ == "__main__":
    sym = net.get_symbol(True)
    num_gpus = 1
    (train_iter, test_iter) = get_data_iterator()
    mod_score = fit(sym, train_iter, test_iter, batch_size, num_gpus)
    
    # assert mod_score > 0.77, "Low training accuracy."
