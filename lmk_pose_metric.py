#mxnet_path = "/home/users/yuxi.feng/Project/mxnet_face/python"                                   
import sys                                                                                       
#sys.path.insert(0, mxnet_path) 
import find_mxnet
import mxnet as mx
import numpy as np
from utils import *

#f = open('/home/users/zhixuan.chen/project/mx_lmk/train.log','wb')

def lmk_transform_func(data):
     data[:, :, 0] = (data[:, :, 0] + 1) * 0.5 * 72
     data[:, :, 1] = (data[:, :, 1] + 1) * 0.5 * 72
     return data 

class LmkMetric(mx.metric.EvalMetric):
    """ 
    This metric is used for lower mxnet version.
    Calculate the mean error of landmarks. The mean error is measured by
    the distances between estimated landmarks and the ground truths, and
    normalized with respect to the inter-ocular distance.
    """

    def __init__(self, lmk_count, pt_ind1, pt_ind2):
        """
        Parameters
        ----------
        lmk_count: landmark count
        pt_ind1/pt_ind2: indexs of points which are used for calculating
            inter-ocular distance.
        # transform_funcs: function that transform landmarks to origin ones,
        #     just like y = a * x + b
        # buffer_size: preserve last buffer_size iteration's output, and output
        #     their average
        """

        assert pt_ind1 < lmk_count, "pt_ind1 should be less than lmk_count"
        assert pt_ind2 < lmk_count, "pt_ind2 should be less than lmk_count"

        self.lmk_count = lmk_count
        self.pt_ind1 = pt_ind1
        self.pt_ind2 = pt_ind2
        self.transform_funcs = lmk_transform_func

        super(LmkMetric, self).__init__('LmkMetric', lmk_count+1)


    def _check_and_get_data(self, label, pred, transform_func=None):
        if isinstance(label, mx.nd.NDArray):
            label = label.asnumpy()[:, :self.lmk_count*2].copy()
        assert isinstance(label, np.ndarray), 'label should be ndarray'

        if isinstance(pred, mx.nd.NDArray):
            pred = pred.asnumpy().copy()
        assert isinstance(pred, np.ndarray), 'pred should be ndarray'

        label = label.reshape(label.shape[0], -1, 2).copy()
        pred = pred.reshape(pred.shape[0], -1, 2).copy()

        assert label.shape == pred.shape, "label and pred should be same size"
        assert label.shape[1] <= self.lmk_count, \
            "pred landmark count should be less than lmk_count"

        if transform_func is not None:
            label = transform_func(label)
            pred = transform_func(pred)
        return (label, pred)


    def update(self, labels, preds):
        label, pred = self._check_and_get_data(labels[0], preds[0], self.transform_funcs)
        num_inst = label.shape[0]
        lmk_cnt = label.shape[1]

        inter_ocular = label[:, self.pt_ind1, :] - label[:, self.pt_ind2, :]
        inter_ocular = np.square(inter_ocular)
        inter_ocular = np.sqrt(inter_ocular[:, 0] + inter_ocular[:, 1])

        error = np.square(label - pred)
        error = np.sqrt(error[:, :, 0] + error[:, :, 1])
        for n in range(num_inst):
            error[n, :] /= (inter_ocular[n])

        # total_error = 0
        for c in range(lmk_cnt):
            # cur_error = np.sum(error[:, c].ravel())
            # total_error += cur_error
            self.num_inst[c + 1] += 1
            self.sum_metric[c + 1] += np.mean(error[:, c])

        # total_error /= lmk_cnt
        self.num_inst[0] += 1
        self.sum_metric[0] += np.mean(error)


class LmkMetric2(mx.metric.EvalMetric):
    """
    This metric is used for latest mxnet version
    """

    def __init__(self, ind, lmk_count, pt_ind1, pt_ind2):


        assert pt_ind1 < lmk_count, "pt_ind1: %d should be less than lmk_count: %d" % (pt_ind1, lmk_count)
        assert pt_ind2 < lmk_count, "pt_ind2 should be less than lmk_count"

        self.lmk_count = lmk_count
        self.pt_ind1 = pt_ind1
        self.pt_ind2 = pt_ind2
        self.transform_funcs = lmk_transform_func
        self.ind = ind

        super(LmkMetric2, self).__init__("LmkMetric_" + str(ind))
    
    def _check_and_get_data(self, label, pred, transform_func=None):
        if isinstance(label, mx.nd.NDArray):
            label = label.asnumpy()[:, :self.lmk_count*2].copy()
        assert isinstance(label, np.ndarray), 'label should be ndarray'

        if isinstance(pred, mx.nd.NDArray):
            pred = pred.asnumpy().copy()
        assert isinstance(pred, np.ndarray), 'pred should be ndarray'

        label = label.reshape(label.shape[0], -1, 2).copy()
        pred = pred.reshape(pred.shape[0], -1, 2).copy()

        assert label.shape == pred.shape, "label and pred should be same size"
        assert label.shape[1] <= self.lmk_count, \
            "pred landmark count should be less than lmk_count"

        if transform_func is not None:
            label = transform_func(label)
            pred = transform_func(pred)
        return (label, pred)

    def update(self,labels,preds):
        label, pred = self._check_and_get_data(labels[0], preds[0], self.transform_funcs)

        num_inst = label.shape[0]
        lmk_cnt = label.shape[1]

        inter_ocular = label[:, self.pt_ind1, :] - label[:, self.pt_ind2, :]
        inter_ocular = np.square(inter_ocular)
        inter_ocular = np.sqrt(inter_ocular[:, 0] + inter_ocular[:, 1])

        error = np.square(label - pred)
        error = np.sqrt(error[:, :, 0] + error[:, :, 1])
        for n in range(num_inst):
            error[n, :] /= (inter_ocular[n])

        self.num_inst += 1
        if self.ind != 0:
            self.sum_metric += np.mean(error[:, self.ind-1])
        else:
            self.sum_metric += np.mean(error)
        #f.write(str(self.sum_metric)+'\n')


class PoseMetric(mx.metric.EvalMetric):
    '''
    order: pitch - yaw - roll
    caculation: rmse for each pose
    '''

    def __init__(self, pose_num):
        self.pose_count = pose_num
        self.transform_func = pose_transform_func
        super(PoseMetric, self).__init__("PoseMetric", pose_num+1)


    def _check_and_get_data(self, labels, preds, transform_func=None):
        if isinstance(labels[0], mx.nd.NDArray):
            label = labels[0].asnumpy()[:, -3:].copy()
        assert isinstance(label, np.ndarray), 'label should be ndarray'

        if isinstance(preds[0], mx.nd.NDArray):
            pred = preds[0].asnumpy().copy()
        assert isinstance(pred, np.ndarray), 'pred should be ndarray'

        assert label.shape == pred.shape, "label and pred should be same size, label = {}, pred = {}".format(label.shape, pred.shape)
        assert label.shape[1] == self.pose_count, \
            "pred pose count should be equal as pose_count"

        if transform_func is not None:
            label = transform_func(label)
            pred = transform_func(pred)
        return (label, pred)


    def update(self, labels, preds):
        label, pred = self._check_and_get_data(labels, preds, self.transform_func)
        num_inst = label.shape[0]
        pose_cnt = label.shape[1]

        error = np.square(label - pred)
        error = np.sqrt(error)
        error = np.mean(error, axis=0)
        for c in range(pose_cnt):
            self.num_inst[c + 1] += 1
            self.sum_metric[c + 1] += error[c]

        self.num_inst[0] += 1
        self.sum_metric[0] += np.mean(error)


class LmkAttrMetric(mx.metric.EvalMetric):
    '''
    caculation: accuracy for each pt attribute
    '''

    def __init__(self, ind):
        # self.pose_count = pose_num
        self.transform_func = None
        self.ind = ind
        super(LmkAttrMetric, self).__init__("LmkAttrMetric_" + str(ind))


    def _check_and_get_data(self, labels, preds, transform_func=None):
        if isinstance(labels[0], mx.nd.NDArray):
            label = labels[0].asnumpy().copy()
            label1 = label[:, 0].ravel()
            label2 = label[:, 1].ravel()
            label3 = label[:, 2].ravel()
            label4 = label[:, 3].ravel()
            label5 = label[:, 4].ravel()
        assert isinstance(label, np.ndarray), 'label should be ndarray'

        if isinstance(preds[0], mx.nd.NDArray):
            pred1 = mx.ndarray.argmax_channel(preds[0]).asnumpy().astype('int32')
            pred2 = mx.ndarray.argmax_channel(preds[1]).asnumpy().astype('int32')
            pred3 = mx.ndarray.argmax_channel(preds[2]).asnumpy().astype('int32')
            pred4 = mx.ndarray.argmax_channel(preds[3]).asnumpy().astype('int32')
            pred5 = mx.ndarray.argmax_channel(preds[4]).asnumpy().astype('int32')

        assert isinstance(pred1, np.ndarray), 'pred1 should be ndarray'
        assert isinstance(pred2, np.ndarray), 'pred2 should be ndarray'
        assert isinstance(pred3, np.ndarray), 'pred3 should be ndarray'
        assert isinstance(pred4, np.ndarray), 'pred4 should be ndarray'
        assert isinstance(pred5, np.ndarray), 'pred5 should be ndarray'

        assert label1.shape[0] == pred1.shape[0], "label and pred should be same size, label = {}, pred = {}".format(label1.shape, pred1.shape)
        assert label2.shape[0] == pred2.shape[0], "label and pred should be same size, label = {}, pred = {}".format(label2.shape, pred2.shape)
        assert label3.shape[0] == pred3.shape[0], "label and pred should be same size, label = {}, pred = {}".format(label3.shape, pred3.shape)
        assert label4.shape[0] == pred4.shape[0], "label and pred should be same size, label = {}, pred = {}".format(label4.shape, pred4.shape)
        assert label5.shape[0] == pred5.shape[0], "label and pred should be same size, label = {}, pred = {}".format(label5.shape, pred5.shape)

        if self.transform_func is not None:
            label = self.transform_func(label)
            pred = self.transform_func(pred)
        return ([label1, label2, label3, label4, label5], [pred1, pred2, pred3, pred4, pred5])


    def update(self, labels, preds):
        labels, preds = self._check_and_get_data(labels, preds)
        # print label, pred

        # num_inst = labels[0].shape[0]
        # error = np.zeros(5)
        # for i in range(5):
        #     error[i] = np.sum(preds[i].flat == labels[i].flat) * 1.0 / num_inst

        # if self.ind != 0:
        #     self.sum_metric += error[self.ind-1]
        #     self.num_inst += 1
        # else:
        #     self.sum_metric += np.mean(error)
        #     self.num_inst += 1

        if self.ind != 0:
            keep_inds = (labels[self.ind-1] == 1)
            num_inst = np.sum(keep_inds)
            self.sum_metric += np.sum(preds[self.ind-1][keep_inds].flat == labels[self.ind-1][keep_inds].flat)
            self.num_inst += num_inst
        else:
            error = 0
            num_inst = 0
            for i in range(5):
                keep_inds = (labels[i] == 1)
                error += np.sum(preds[i][keep_inds].flat == labels[i][keep_inds].flat)
                num_inst += np.sum(keep_inds)
            self.sum_metric += error
            self.num_inst += num_inst
            
