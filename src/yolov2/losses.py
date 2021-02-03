import numpy as np
import tensorflow as tf


def get_cell_grid(GRID_W, GRID_H, BATCH_SIZE, BOX):
    """
    Helper function to assure that the bounding box x and y are in the grid cell scale
    == output ==
    for any i=0,1..,batch size - 1
    output[i,5,3,:,:] = array([[3., 5.],
                               [3., 5.],
                               [3., 5.]], dtype=float32)
    """
    # cell_x.shape = (1, 13, 13, 1, 1)
    # cell_x[:,i,j,:] = [[[j]]]
    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)

    # cell_y.shape = (1, 13, 13, 1, 1)
    # cell_y[:,i,j,:] = [[[i]]]
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    # cell_gird.shape = (16, 13, 13, 5, 2)
    # for any n, k, i, j
    #    cell_grid[n, i, j, anchor, k] = j when k = 0
    # for any n, k, i, j
    #    cell_grid[n, i, j, anchor, k] = i when k = 1
    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, BOX, 1])
    return cell_grid


def adjust_scale_prediction(y_pred, cell_grid, anchors):
    """
        Adjust prediction

        == input ==

        y_pred : takes any real values
                 tensor of shape = (N batch, NGrid h, NGrid w, NAnchor, 4 + 1 + N class)

        ANCHORS : list containing width and height specialization of anchor box
        == output ==

        pred_box_xy : shape = (N batch, N grid x, N grid y, N anchor, 2), containing [center_y, center_x]
                                ranging [0,0]x[grid_H-1,grid_W-1]

          pred_box_xy[i_row, i_grid_h, i_grid_w, i_anchor, 0] = center_x
          pred_box_xy[i_row, i_grid_h, i_grid_w, i_anchor, 1] = center_1

          calculation process:
          tf.sigmoid(y_pred[...,:2]) : takes values between 0 and 1
          tf.sigmoid(y_pred[...,:2]) + cell_grid : takes values between 0 and grid_W - 1 for x coordinate
                                                   takes values between 0 and grid_H - 1 for y coordinate

        pred_Box_wh : shape = (N batch, N grid h, N grid w, N anchor, 2), containing width and height,
                                ranging [0,0]x[grid_H-1,grid_W-1]

        pred_box_conf : shape = (N batch, N grid h, N grid w, N anchor, 1),
                                containing confidence to range between 0 and 1

        pred_box_class : shape = (N batch, N grid h, N grid w, N anchor, N class), containing
    """

    box = int(len(anchors) / 2)
    # cell_grid is of the shape of

    # adjust x and y
    # the bounding box bx and by are rescaled to range between 0 and 1 for given gird.
    # Since there are BOX x BOX grids, we rescale each bx and by to range between 0 to BOX + 1
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid  # bx, by

    # adjust w and h
    # exp to make width and height positive
    # rescale each grid to make some anchor "good" at representing certain shape of bounding box
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(anchors, [1, 1, 1, box, 2])  # bw, bh

    # adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])  # prob bb

    # adjust class probabilities
    pred_box_class = y_pred[..., 5:]  # prC1, prC2, ..., prC20

    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class


def print_min_max(vec, title):
    try:
        print("{} MIN={:5.2f}, MAX={:5.2f}".format(
            title, np.min(vec), np.max(vec)))

    # Raised if `y` is empty.
    except ValueError:
        pass


def extract_ground_truth(y_true):
    true_box_xy = y_true[..., 0:2]  # bounding box x, y coordinate in grid cell scale
    true_box_wh = y_true[..., 2:4]  # number of cells across, horizontally and vertically
    true_box_conf = y_true[..., 4]  # confidence
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    return true_box_xy, true_box_wh, true_box_conf, true_box_class


def calc_loss_xywh(true_box_conf, COORD_SCALE, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh, lambda_coord):
    coord_mask = tf.expand_dims(true_box_conf, axis=-1) * lambda_coord
    nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, tf.float32))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.

    return loss_xy + loss_wh, coord_mask


def calc_loss_class(true_box_conf, CLASS_SCALE, true_box_class, pred_box_class):
    """
    == input ==
    true_box_conf  : tensor of shape (N batch, N grid h, N grid w, N anchor)
    true_box_class : tensor of shape (N batch, N grid h, N grid w, N anchor), containing class index
    pred_box_class : tensor of shape (N batch, N grid h, N grid w, N anchor, N class)
    CLASS_SCALE    : 1.0

    == output ==
    class_mask
    if object exists in this (grid_cell, anchor) pair and the class object receive nonzero weight
        class_mask[iframe, igridy, igridx, ianchor] = 1
    else:
        0
    """
    class_mask = true_box_conf * CLASS_SCALE  # L_{i,j}^obj * lambda_class

    nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class,
                                                                logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    return loss_class


def get_intersect_area(true_xy, true_wh,
                       pred_xy, pred_wh):
    '''
    == INPUT ==
    true_xy,pred_xy, true_wh and pred_wh must have the same shape length

    p1 : pred_mins = (px1,py1)
    p2 : pred_maxs = (px2,py2)
    t1 : true_mins = (tx1,ty1)
    t2 : true_maxs = (tx2,ty2)
                 p1______________________
                 |      t1___________   |
                 |       |           |  |
                 |_______|___________|__|p2
                         |           |rmax
                         |___________|
                                      t2
    intersect_mins : rmin = t1  = (tx1,ty1)
    intersect_maxs : rmax = (rmaxx,rmaxy)
    intersect_wh   : (rmaxx - tx1, rmaxy - ty1)

    '''
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)
    return iou_scores


def calc_IOU_pred_true_assigned(true_box_conf,
                                true_box_xy, true_box_wh,
                                pred_box_xy, pred_box_wh):
    """
    == input ==

    true_box_conf : tensor of shape (N batch, N grid h, N grid w, N anchor )
    true_box_xy   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
    true_box_wh   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
    pred_box_xy   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)
    pred_box_wh   : tensor of shape (N batch, N grid h, N grid w, N anchor , 2)

    == output ==

    true_box_conf : tensor of shape (N batch, N grid h, N grid w, N anchor)

    true_box_conf value depends on the predicted values
    true_box_conf = IOU_{true,pred} if objects exist in this anchor else 0
    """
    iou_scores = get_intersect_area(true_box_xy, true_box_wh,
                                    pred_box_xy, pred_box_wh)
    true_box_conf_IOU = iou_scores * true_box_conf
    return true_box_conf_IOU


def calc_IOU_pred_true_best(pred_box_xy, pred_box_wh, true_boxes):
    """
    == input ==
    pred_box_xy : tensor of shape (N batch, N grid h, N grid w, N anchor, 2)
    pred_box_wh : tensor of shape (N batch, N grid h, N grid w, N anchor, 2)
    true_boxes  : tensor of shape (N batch, N grid h, N grid w, N anchor, 2)

    == output ==

    best_ious

    for each iframe,
        best_ious[iframe, igridy, igridx, ianchor] contains

        the IOU of the object that is most likely included (or best fitted)
        within the bounded box recorded in (grid_cell, anchor) pair

        NOTE: a same object may be contained in multiple (grid_cell, anchor) pair
              from best_ious, you cannot tell how may actual objects are captured as the "best" object
    """
    true_xy = true_boxes[..., 0:2]  # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)
    true_wh = true_boxes[..., 2:4]  # (N batch, 1, 1, 1, TRUE_BOX_BUFFER, 2)

    pred_xy = tf.expand_dims(pred_box_xy, 4)  # (N batch, N grid_h, N grid_w, N anchor, 1, 2)
    pred_wh = tf.expand_dims(pred_box_wh, 4)  # (N batch, N grid_h, N grid_w, N anchor, 1, 2)

    iou_scores = get_intersect_area(true_xy,
                                    true_wh,
                                    pred_xy,
                                    pred_wh)  # (N batch, N grid_h, N grid_w, N anchor, 50)

    best_ious = tf.reduce_max(iou_scores, axis=4)  # (N batch, N grid_h, N grid_w, N anchor)
    return best_ious


def get_conf_mask(best_ious, true_box_conf, true_box_conf_IOU, LAMBDA_NO_OBJECT, LAMBDA_OBJECT):
    """
    == input ==

    best_ious           : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    true_box_conf       : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    true_box_conf_IOU   : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    LAMBDA_NO_OBJECT    : 1.0
    LAMBDA_OBJECT       : 5.0

    == output ==
    conf_mask : tensor of shape (Nbatch, N grid h, N grid w, N anchor)

    conf_mask[iframe, igridy, igridx, ianchor] = 0
               when there is no object assigned in (grid cell, anchor) pair and the region seems useless i.e.
               y_true[iframe,igridx,igridy,4] = 0 "and" the predicted region has no object that has IoU > 0.6

    conf_mask[iframe, igridy, igridx, ianchor] =  NO_OBJECT_SCALE
               when there is no object assigned in (grid cell, anchor) pair but region seems to include some object
               y_true[iframe,igridx,igridy,4] = 0 "and" the predicted region has some object that has IoU > 0.6

    conf_mask[iframe, igridy, igridx, ianchor] =  OBJECT_SCALE
              when there is an object in (grid cell, anchor) pair
    """

    conf_mask = tf.cast(best_ious < 0.6, tf.float32) * (1 - true_box_conf) * LAMBDA_NO_OBJECT
    # penalize the confidence of the boxes, which are responsible for corresponding ground truth box
    conf_mask = conf_mask + true_box_conf_IOU * LAMBDA_OBJECT
    return conf_mask


def calc_loss_conf(conf_mask, true_box_conf_IOU, pred_box_conf):
    """
    == input ==

    conf_mask         : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    true_box_conf_IOU : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    pred_box_conf     : tensor of shape (Nbatch, N grid h, N grid w, N anchor)
    """
    # the number of (grid cell, anchor) pair that has an assigned object or
    # that has no assigned object but some objects may be in bounding box.
    # N conf
    nb_conf_box = tf.reduce_sum(tf.cast(conf_mask > 0.0, tf.float32))
    loss_conf = tf.reduce_sum(tf.square(true_box_conf_IOU - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    return loss_conf
