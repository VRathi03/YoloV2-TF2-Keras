import numpy as np


class BestAnchorBoxFinder(object):
    def __init__(self, anchors):
        """
        ANCHORS: a np.array of even number length e.g.

        _ANCHORS = [4,2, #  width=4, height=2,  flat large anchor box
                    2,4, #  width=2, height=4,  tall large anchor box
                    1,1] #  width=1, height=1,  small anchor box
        """
        self.anchors = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1])
                        for i in range(int(len(anchors) // 2))]

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

        union = w1 * h1 + w2 * h2 - intersect

        return float(intersect) / union

    def find(self, center_w, center_h):
        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou = -1
        # each Anchor box is specialized to have a certain shape.
        # e.g., flat large rectangle, or small square
        shifted_box = BoundBox(0, 0, center_w, center_h)
        # For given object, find the best anchor box!
        for i in range(len(self.anchors)):  # run through each anchor box
            anchor = self.anchors[i]
            iou = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou = iou
        return best_anchor, max_iou


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax

        # the code below are used during inference probability
        self.confidence = confidence

        # class probabilities [c1, c2, .. cNClass]
        self.set_class(classes)

    def set_class(self, classes):
        # TODO move to init
        self.classes = classes
        self.label = np.argmax(self.classes)

    def get_label(self):
        return self.label

    def get_score(self):
        return self.classes[self.label]


def rescale_center_xy(obj, config):
    """
    obj:     dictionary containing xmin, xmax, ymin, ymax
    config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
    """

    center_x = .5 * (obj['xmin'] + obj['xmax'])
    center_x = center_x / (float(config['image_w']) / config['grid_w'])

    center_y = .5 * (obj['ymin'] + obj['ymax'])
    center_y = center_y / (float(config['image_h']) / config['grid_h'])

    return center_x, center_y


def rescale_center_wh(obj, config):
    """
    obj:     dictionary containing x_min, x_max, y_min, y_max
    config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
    """

    # unit: grid cell
    center_w = (obj['xmax'] - obj['xmin']) / (float(config['image_w']) / config['grid_w'])

    # unit: grid cell
    center_h = (obj['ymax'] - obj['ymin']) / (float(config['image_h']) / config['grid_h'])

    return center_w, center_h
