#
# Class for labelme json management
#
import json
import os, cv2, copy

from .polygon import Polygon

class Labelme(object):
    def __init__(self, image_info, shapes):
        self.shape_list = shapes
        self.image_info = image_info
        self.image_path = image_info['image_path']
        self.image_height = image_info['height']
        self.image_width = image_info['width']

    @classmethod
    def from_json(cls, labelme_json):
        data = cls.parse_json(labelme_json)
        return cls(image_info=data['image_info'], shapes=data['shapes'])

    def __len__(self):
        return len(self.shape_list)

    def iterator(self):
        for shape in self.shape_list:
            yield shape

    def clone(self):
        return copy.deepcopy(self)

    @staticmethod
    def parse_json(labelme_json):
        try:
            with open(labelme_json, encoding='gbk') as fid:
                obj = json.load(fid)
        except:
            with open(labelme_json, encoding='utf-8') as fid:
                obj = json.load(fid)

        # get shapes
        shapes = list()
        for shape in obj['shapes']:
            polygon = Polygon(
                points=shape['points'],
                class_name=shape['label'],
                shape_type=shape['shape_type']
            )
            shapes.append(polygon)

        image_info = {
            'image_path': obj['imagePath'].replace('\\', '/'),  # path to linux style
            'height': obj['imageHeight'], 'width': obj['imageWidth']
        }
        data = {'shapes': shapes, 'image_info':  image_info}

        return data

    def load_image(self, image_root=''):
        image_path = os.path.join(image_root, self.image_path)
        assert os.path.exists(image_path), "Cannot find image: {}".format(image_path)
        image_bgr = cv2.imread(image_path)
        image_rgb = copy.deepcopy(image_bgr[...,::-1]) # deepcopy to get around the polyline drawing error
        return image_rgb

    def draw_shapes(self, image):
        """ draw shapes on image """
        for shape in self.iterator():
            image = shape.draw_on(image)
        return image
