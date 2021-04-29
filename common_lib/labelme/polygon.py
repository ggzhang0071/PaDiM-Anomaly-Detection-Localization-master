#
# Definition of Polygon class
#
import copy, cv2
import numpy as np

class Polygon(object):
    def __init__(self, points, class_name, shape_type):
        assert isinstance(points, list) or isinstance(points, tuple)
        assert isinstance(class_name, str)
        self.class_name = class_name
        self.points = self.shape_as_polygon(points, shape_type)
        self.shape_type = 'polygon'

    def shape_as_polygon(self, points, shape_type):
        assert shape_type in ['polygon', 'rectangle', 'circle']
        if None: pass
        elif shape_type == 'polygon':
            return points
        elif shape_type == 'rectangle':
            x1,y1 = points[0]
            x2,y2 = points[1]
            return [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
        elif shape_type == 'circle':
            x0,y0 = points[0]
            x1,y1 = points[1]
            r = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            angles = np.linspace(0, 2*np.pi, 36)
            Xs, Ys = x0+r*np.cos(angles), y0+r*np.sin(angles)
            points = [[float(x),float(y)] for x,y in zip(Xs,Ys)]
            return points
        else:
            raise NotImplementedError("Unrecognized shape type: {}".format(self.shape_type))

    def clone(self):
        return copy.deepcopy(self)

    def draw_on(self, image, linewidth=20, color=(255,0,0)):
        image_with_polygon = cv2.polylines(image, np.array([self.points], np.int32), True, color=color, thickness=linewidth)
        return image_with_polygon

    def set_class(self, class_name):
        assert isinstance(class_name, str) 
        self.class_name = class_name

    def json_format(self):
        """ prepare for labelme json dumping """
        shape_obj = {
            'label': self.class_name,
            'points': self.points,
            'shape_type': self.shape_type,
        }
        return shape_obj

    def __eq__(self, other):
        if self.class_name == other.class_name \
                and self.points == other.points \
                and self.shape_type == other.shape_type:
            return True
        else:
            return False
