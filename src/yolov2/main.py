import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence

from config import parser
from data_reader import parse_annotation, ImageReader
from losses import get_cell_grid, adjust_scale_prediction, print_min_max
from model import build_model
from utils import get_project_root, set_pre_trained_weights, initialize_weights
from yolo_backend import BestAnchorBoxFinder, rescale_center_wh, rescale_center_xy

np.random.seed(1)
base_path = get_project_root()

""" ################  SECTION 1  ####################### """
train_image_dir = os.path.join(base_path, parser.get('data', 'train_images'))
train_annotations_dir = os.path.join(base_path, parser.get('data', 'train_annotations'))
training_labels = parser.get('data', 'labels')

train_image, seen_train_labels = parse_annotation(train_annotations_dir,
                                                  train_image_dir,
                                                  labels=training_labels)
print("N train = {}".format(len(train_image)))

""" ################  SECTION 2  ####################### """


# Helper Functions that use Keras, importing Keras at multiple locations resulting in memory errors


class CustomLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (0, 0.01),
    (75, 0.001),
    (105, 0.0001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


class SimpleBatchGenerator(Sequence):
    def __init__(self, images, config, norm=None, shuffle=True):
        """
        config : dictionary containing necessary hyper parameters for training. e.g.,
            {
            'image_h'         : 416,
            'image_w'         : 416,
            'grid_w'          : 13,
            'grid_h'          : 13,
            'labels'          : ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle',
                                  'bus',        'car',      'cat',  'chair',     'cow',
                                  'dining table','dog',    'horse',  'motorbike', 'person',
                                  'potted plant','sheep',  'sofa',   'train',   'tv monitor'],
            'anchors'         : array([ 1.07709888,   1.78171903,
                                        2.71054693,   5.12469308,
                                        10.47181473, 10.09646365,
                                        5.48531347,   8.11011331]),
            'batch_size'      : 16,
            'true_box_buffer' : 50,
            }

        """
        self.config = config
        self.config["box"] = int(len(self.config['anchors']) / 2)
        self.config["class"] = len(self.config['labels'])
        self.images = images
        self.bestAnchorBoxFinder = BestAnchorBoxFinder(config['anchors'])
        self.imageReader = ImageReader(config['image_h'], config['image_w'], norm=norm)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config['batch_size']))

    def __getitem__(self, idx):
        """
        == input ==

        idx : non-negative integer value e.g., 0

        == output ==

        x_batch: The numpy array of shape  (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels).

            x_batch[iframe,:,:,:] contains a i_frame_th frame of size  (IMAGE_H,IMAGE_W).

        y_batch:

            The numpy array of shape  (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes).
            BOX = The number of anchor boxes.

            y_batch[iframe, i_grid_h, i_grid_w, i_anchor,:4] contains (center_x,center_y,center_w,center_h)
            of i_anchor_th anchor at  grid cell=(i_grid_h, i_grid_w) if the object exists in
            this (grid cell, anchor) pair, else they simply contain 0.

            y_batch[iframe, i_grid_h, i_grid_w, i_anchor,4] contains 1 if the object exists in this
            (grid cell, anchor) pair, else it contains 0.

            y_batch[iframe, i_grid_h, i_grid_w, i_anchor, 5 + i_class] contains 1 if the iclass^th
            class object exists in this (grid cell, anchor) pair, else it contains 0.


        b_batch:

            The numpy array of shape (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4).

            b_batch[iframe, 1, 1, 1, i_buffer, i_anchor, :] contains i_buffer_th object's
            (center_x, center_y, center_w, center_h) in i_frame_th frame.

            If i_buffer > N objects in i_frame_th frame, then the values are simply 0.

            TRUE_BOX_BUFFER has to be some large number, so that the frame with the
            biggest number of objects can also record all objects.

            The order of the objects do not matter.

            This is just a hack to easily calculate loss.

        """
        l_bound = idx * self.config['batch_size']
        r_bound = (idx + 1) * self.config['batch_size']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['batch_size']

        instance_count = 0

        # prepare empty storage space: this will be output

        # input images
        x_batch = np.zeros((r_bound - l_bound, self.config['image_h'], self.config['image_w'], 3))

        # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['true_box_buffer'], 4))

        # desired network output
        y_batch = np.zeros((r_bound - l_bound, self.config['grid_h'], self.config['grid_w'], self.config['box'],
                            4 + 1 + len(self.config['labels'])))

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.imageReader.fit(train_instance)

            # construct output from object's x, y, w, h
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['labels']:
                    center_x, center_y = rescale_center_xy(obj, self.config)

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['grid_w'] and grid_y < self.config['grid_h']:
                        obj_index = self.config['labels'].index(obj['name'])
                        center_w, center_h = rescale_center_wh(obj, self.config)
                        box = [center_x, center_y, center_w, center_h]
                        best_anchor, max_iou = self.bestAnchorBoxFinder.find(center_w, center_h)

                        # assign ground truth x, y, w, h, confidence and class probabilities to y_batch
                        # it could happen that the same grid cell contain 2 similar shape objects
                        # as a result the same anchor box is selected as the best anchor box by the multiple objects
                        # in such ase, the object is over written
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box  # center_x, center_y, w, h
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.  # ground truth confidence is 1

                        # class probability of the object
                        y_batch[
                            instance_count, grid_y, grid_x, best_anchor, 5 + obj_index] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['true_box_buffer']

            x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)


""" ################  SECTION 3 - LOSS  ####################### """
# Loss Functions

""" ################  SECTION 4  ####################### """
# TODO Sort this from config file
anchors = np.array([0.08285376, 0.13705531,
                    0.20850361, 0.39420716,
                    0.80552421, 0.77665105,
                    0.42194719, 0.62385487])

grid_h, grid_w = 13, 13
anchors[::2] = anchors[::2] * grid_w
anchors[1::2] = anchors[1::2] * grid_h

image_h, image_w = 416, 416
batch_size = 2
true_box_buffer = 50
box = int(len(anchors) / 2)
classes = len(training_labels)

generator_config = {
    'image_h': image_h,
    'image_w': image_w,
    'grid_h': grid_h,
    'grid_w': grid_w,
    'box': box,
    'labels': training_labels,
    'anchors': anchors,
    'batch_size': batch_size,
    'true_box_buffer': true_box_buffer
}

""" ################  SECTION 4  ####################### """
# Create Model and Print Model Architecture
model = build_model(image_h, image_w, grid_h, grid_w, box, classes, true_box_buffer)

# To Print Model Summary on Console
# print(model.summary())

""" ################  SECTION 4  ####################### """
# Weight Initialization

nb_conv = int(parser.get('model', 'number_of_conv_layers'))
path_to_weights = os.path.join(base_path, parser.get('model', 'pre_trained_weights_path'))

model = set_pre_trained_weights(model, nb_conv, path_to_weights)

# Initialize the final convolutional layer
# layer = model.layers[-4]  # the last convolutional layer
# initialize_weights(layer, sd=grid_h * grid_w)

""" ############################# TESTING AREA ################################### """

# print("*" * 30)
# print("Prepare Inputs")
#
# size = batch_size * grid_w * grid_h * box * (4 + 1 + classes)
#
# y_pred = np.random.normal(size=size, scale=10 / (grid_h * grid_w))
# y_pred = y_pred.reshape(batch_size, grid_h, grid_w, box, 4 + 1 + classes)
#
# print("y_pred before scaling = {}".format(y_pred.shape))
#
# print("*" * 30)
# print("Define Tensor Graph")
# y_pred_tf = tf.constant(y_pred, dtype="float32")
# cell_grid = get_cell_grid(grid_w, grid_h, batch_size, box)
# (pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class) = adjust_scale_prediction(y_pred_tf,
#                                                                                     cell_grid,
#                                                                                     anchors)
# print("*" * 30 + "\noutput\n" + "*" * 30)
#
# print("\npred_box_xy {}".format(pred_box_xy.shape))
#
# for igrid_w in range(pred_box_xy.shape[2]):
#     print_min_max(pred_box_xy[:, :, igrid_w, :, 0],
#                   "  bounding box x at iGRID_W={:02.0f}".format(igrid_w))
# for igrid_h in range(pred_box_xy.shape[1]):
#     print_min_max(pred_box_xy[:, igrid_h, :, :, 1],
#                   "  bounding box y at iGRID_H={:02.0f}".format(igrid_h))
#
# print("\npred_box_wh {}".format(pred_box_wh.shape))
# print_min_max(pred_box_wh[:, :, :, :, 0], "  bounding box width ")
# print_min_max(pred_box_wh[:, :, :, :, 1], "  bounding box height")
#
# print("\npred_box_conf {}".format(pred_box_conf.shape))
# print_min_max(pred_box_conf, "  confidence ")

# print("\npred_box_class {}".format(pred_box_class.shape))
# print_min_max(pred_box_class, "  class probability")
