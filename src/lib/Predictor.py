# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from collections import namedtuple
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

from .Yolo import YOLO

PredictResultItem = namedtuple('PredictResultItem', ('class_index', 'class_label', 'left', 'top', 'width', 'height'))

class Predictor:
    _instance = None
    _model = None
    _class_num = 0
    _labels = []

    def __init__(self):
        pass

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path, anchors_path, classes_path):
        model = YOLO(model_path=model_path, anchors_path=anchors_path, classes_path=classes_path)
        model.yolo_model.summary()
        self._model = model
    
    def predict(self, image):
        boxes = self._model.detect_image(image)
        print(boxes)
        resutls = []
        for class_index, predicted_class, score, left, top, right, bottom in boxes:
            resutls.append(PredictResultItem(
                class_index=class_index,
                class_label=predicted_class,
                left=left,
                top=top,
                width=right,
                height=bottom
            ))
        return resutls


#     def predict(self, image):
#         """
#         image : PIL image
#         """
#         if self._model == None:
#             raise Exception("illegal operation. model is not load yet.")

#         input_image, resize_sclae = self._resize_image(image, 416)
#         input_image = img_to_array(input_image)
#         input_image = np.expand_dims(input_image, axis=0)
#         input_image /= 255.

#         res = self._model.predict(input_image)

# #        scale_x = 1 / resize_sclae
# #        scale_y = 1 / resize_sclae
#         scale_x = image.size[0] / 416
#         scale_y = image.size[1] / 416
#         print(image.size, scale_x, scale_y)
#         boxes = self._parse_yolo_out(res, scale_x, scale_y)

#         return boxes

    def _resize_image(self, image, target_size=416):
        scale = target_size / max(image.size)
        # new_width = int(image.size[0] * scale)
        # new_height = int(image.size[1] * scale)
        # image = image.resize((new_width, new_height), Image.LANCZOS)
        image = image.resize((target_size, target_size), Image.LANCZOS)
        return image, scale

    def _parse_yolo_out(self, model_result, scale_x=0, scale_y=0):
        """ 整形する, ラベルを付与する """
        yolo = TinyYolo(classes=self._class_num)
        boxes = yolo.parse_result(model_result) # [ { xmin, ymin, xmax, ymax, class, prob } ]
        resutls = []
        for box in boxes:
            xmin = int(box["xmin"] * scale_x)
            ymin = int(box["ymin"] * scale_y)
            xmax = int(box["xmax"] * scale_x)
            ymax = int(box["ymax"] * scale_y)
            class_index = int(box["class"])
            class_label = self._labels[class_index]
            resutls.append(PredictResultItem(
                class_index=class_index,
                class_label=class_label,
                left=xmin,
                top=ymin,
                width=xmax-xmin,
                height=ymax-ymin
            ))
        return resutls