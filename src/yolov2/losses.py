# import tensorflow as tf
import numpy as np


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
