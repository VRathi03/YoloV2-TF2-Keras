import copy
import os
import xml.etree.ElementTree as ET

import cv2


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_images = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        if "xml" not in ann:
            continue
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                path_to_image = img_dir + elem.text
                img['filename'] = path_to_image
                # make sure that the image exists:
                if not os.path.exists(path_to_image):
                    assert False, "file does not exist!\n{}".format(path_to_image)
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_images += [img]

    return all_images, seen_labels


class ImageReader(object):
    def __init__(self, image_h, image_w, norm=None):
        self.image_h = image_h
        self.image_w = image_w
        self.norm = norm

    def encode_core(self, image, reorder_rgb=True):
        image = cv2.resize(image, (self.image_h, self.image_w))
        if reorder_rgb:
            image = image[:, :, ::-1]
        if self.norm is not None:
            image = self.norm(image)
        return image

    def fit(self, train_instance):
        """
        read in and resize the image, annotations are resized accordingly.

        -- Input --

        train_instance : dictionary containing filename, height, width and object

        {'filename': 'VOC2012/JPEGImages/2008_000054.jpg',
         'height':   333,
         'width':    500,
         'object': [{'name': 'bird',
                     'xmax': 318,
                     'xmin': 284,
                     'ymax': 184,
                     'ymin': 100},
                    {'name': 'bird',
                     'xmax': 198,
                     'xmin': 112,
                     'ymax': 209,
                     'ymin': 146}]
        }

        """
        if not isinstance(train_instance, dict):
            train_instance = {'filename': train_instance}

        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        h, w, c = image.shape
        if image is None: print('Cannot find ', image_name)

        image = self.encode_core(image, reorder_rgb=True)

        if "object" in train_instance.keys():

            all_objs = copy.deepcopy(train_instance['object'])

            # fix object's position and size
            for obj in all_objs:
                for attr in ['xmin', 'xmax']:
                    obj[attr] = int(obj[attr] * float(self.image_w) / w)
                    obj[attr] = max(min(obj[attr], self.image_w), 0)

                for attr in ['ymin', 'ymax']:
                    obj[attr] = int(obj[attr] * float(self.image_h) / h)
                    obj[attr] = max(min(obj[attr], self.image_h), 0)
        else:
            return image
        return image, all_objs
