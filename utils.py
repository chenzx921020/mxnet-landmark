import numpy as np
import copy as cp
import cv2, os
import find_mxnet
import mxnet as mx


class Utils(object):
    """ class utils """

    def __init__(self, norm_len, input_w, input_h):
        self.norm_len = norm_len
        self.input_w = input_w
        self.input_h = input_h

    def rectify_bbox(self, bbox):
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]

        length = np.sqrt(height**2 + width**2)
        enlarge_factor = length / self.norm_len

        half_w = self.input_w * enlarge_factor * 0.5
        half_h = self.input_h * enlarge_factor * 0.5

        center_y = (bbox[3] + bbox[1]) / 2
        center_x = (bbox[2] + bbox[0]) / 2

        out_bbox = [-1] * 4
        out_bbox[0] = center_x - half_w
        out_bbox[1] = center_y - half_h
        out_bbox[2] = center_x + half_w
        out_bbox[3] = center_y + half_h
        return out_bbox

    def crop_and_resize_patch(self, src, coords, is_fill=True, fill_value=0):
        x1 = int(min(src.shape[1] - 1, max(0, coords[0])))
        y1 = int(min(src.shape[0] - 1, max(0, coords[1])))
        x2 = int(min(src.shape[1], max(0, coords[2])))
        y2 = int(min(src.shape[0], max(0, coords[3])))

        dst_tmp_h_beg = 0
        dst_tmp_w_beg = 0
        if is_fill:
            crop_width = int(max(1, coords[2] - coords[0]))
            crop_height = int(max(1, coords[3] - coords[1]))
            if coords[0] < 0:
                dst_tmp_w_beg = int(-coords[0])
            if coords[1] < 0:
                dst_tmp_h_beg = int(-coords[1])

        dst_tmp = np.zeros((crop_height, crop_width, 3), dtype=np.uint8) * fill_value

        diff = (crop_width - dst_tmp_w_beg) - (x2 - x1)
        if diff < 0:
            x2 -= -diff
        if diff > 0:
            x2 += diff
        x2 = min(src.shape[1], x2)
        crop_width = dst_tmp_w_beg + x2 - x1
        # print (x1, x2, dst_tmp_w_beg, crop_width)

        for crop_y1, h in zip(range(y1, y2), range(dst_tmp_h_beg, crop_height)):
            dst_tmp[h, dst_tmp_w_beg : crop_width, :] = src[crop_y1, x1 : x2, :]
        res = cv2.resize(dst_tmp, (self.input_h, self.input_w))
        return res

    def project_lmk(self, bbox, lmk):
        """ scale landmark coords as input size: (inputw, inputh) """

        lmk_cnt = lmk.shape[0]
        out_lmk = [0.] * len(lmk) * 2
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        for i in range(lmk_cnt):
            out_lmk[2 * i] = (lmk[i][0] - bbox[0]) * self.input_w / bbox_width
            out_lmk[2 * i + 1] = (lmk[i][1] - bbox[1]) * self.input_h / bbox_height
        return self.norm_lmk(out_lmk)

    def norm_lmk(self, lmk):
        """ (2 * x - width) / width to scale it into [-1, 1] """

        out_lmk = [0.] * len(lmk)
        lmk_cnt = len(lmk) / 2
        for i in range(lmk_cnt):
            out_lmk[2 * i] = (2 * lmk[2 * i] - self.input_w) / self.input_w
            out_lmk[2 * i + 1] = (2 * lmk[2 * i + 1] - self.input_h) / self.input_h
        return np.reshape(np.array(out_lmk), (lmk_cnt, 2))

    def reproject_lmk(self, lmk):
        """ rescale landmark coords back """

        lmk_cnt = lmk.shape[0]
        out_lmk = [0.] * lmk_cnt * 2
        for i in range(lmk_cnt):
            out_lmk[2 * i] = (lmk[i][0] + 1) * self.input_w * 0.5
            out_lmk[2 * i + 1] = (lmk[i][1] + 1) * self.input_h * 0.5
        return np.reshape(np.array(out_lmk), (lmk_cnt, 2))

    def visualize(self, imdb, img_dir, count=0):
        if count != 0:
            vis_count = count
        else:
            vis_count = len(imdb)

        for ind in range(vis_count):
            img_path = os.path.join(img_dir, imdb[ind]["img_path"])
            print(img_path)
            img = cv2.imread(img_path)
            bbox = cp.deepcopy(imdb[ind]["bbox"])
            bbox = map(int, map(float, bbox))
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            lmk = imdb[ind]["landmark_point"].copy()
            lmk_count = lmk.shape[0]
            for i in range(lmk_count):
                pt1 = int(lmk[i][0])
                pt2 = int(lmk[i][1])
                cv2.circle(img, (pt1, pt2), 2, (0, 255, 0), -1)

            vis_dir = os.path.join("visual", imdb[ind]["img_path"].split("/")[0])
            print(vis_dir)
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            save_path = os.path.join("visual", imdb[ind]["img_path"])
            cv2.imwrite(save_path, img)
            vis_count -= 1

    def show_rec_file(self, rec_file):
        record = mx.recordio.MXRecordIO(rec_file, 'r')
        for i in range(100):
            item = record.read()
            header, img = mx.recordio.unpack_img(item)
            print(img.shape)
            flag, label, id1, id2 = header
            lmk_pt = self.reproject_lmk(np.array(label[:42]).reshape((-1, 2))).astype("int")
            lmk_attr = np.array(label[42:63])
            pose = np.array(label[-3:])
            print(lmk_attr)
            print
            print(pose)
            print
            print
            for i in range(lmk_pt.shape[0]):
                cv2.circle(img, (lmk_pt[i][0], lmk_pt[i][1]), 2, (255, 0, 0), -1)
            cv2.imshow("image", img)
            key = chr(cv2.waitKey() % 256)
            if key in ['q', 'Q']:
                break


norm_len = 80
input_w = 72
input_h = 72


if __name__ == "__main__":
    Dataer = Utils(norm_len, input_w, input_h)
    import sys
    rec = sys.argv[1]
    Dataer.show_rec_file(rec)

