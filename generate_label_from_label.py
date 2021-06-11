import numpy as np
import cv2
import math
import random
from enum import Enum
import matplotlib.pyplot as plt


dataset_info = {
    'center': (1500, 1500),
    'radius_outside': 1321,
    'radius_inside': 1057,
    'thresholds': {   # determine whether a crop is NG/OK/Difficult (rejected)
        'd/D': .3,
        'i/d': 3/8
    }
}


class GridCrop:

    circle_info = dataset_info

    class side(Enum):
        outside = 'outside'
        inside = 'inside'

    @classmethod
    def mk_grid_roi(cls, image_width, image_height, crop_width, crop_height, group_num, visual=False):
        grid_crop_obj = cls(image_width, image_height)
        roi_list = grid_crop_obj.get_grid_crop(crop_width, crop_height, group_num)

        if visual:
            image = np.zeros((image_height, image_width, 3), dtype='uint8')
            cv2.circle(image, cls.circle_info['center'], cls.circle_info['radius_outside'], [150, 150, 150], 3)
            cv2.circle(image, cls.circle_info['center'], cls.circle_info['radius_inside'], [150, 150, 150], 3)
            cv2.circle(image, cls.circle_info['center'], 2, [150, 150, 150], 3)

            for roi in roi_list:
                cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), [0, 255, 0], 2)

            plt.imshow(image)
            plt.show()

        return roi_list

    def __init__(self, image_width, image_height):
        assert image_width == image_height == 3000
        self.image_width = image_width
        self.image_height = image_height

    @staticmethod
    def distance_cal(point_1, point_2):
        return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5

    def _crop_data_check(self, crop_width, crop_height):
        crop_width = int(crop_width)
        crop_height = int(crop_height)

        assert crop_width > 0 and crop_height > 0
        return crop_width, crop_height

    def get_grid_angle_by_side(self, crop_width, side, group_num=1):

        def get_grid_angle(step_radian, start_theta):
            grid_num = int(np.ceil(2 * math.pi / step_radian))
            return (np.array(range(grid_num)) * step_radian + start_theta).tolist()

        group_num = max(1, int(group_num))

        assert side in self.side
        radius = self.circle_info['radius_inside'] if side == self.side.inside else self.circle_info['radius_outside']

        step_radian = crop_width / radius
        start_theta_list = (step_radian * np.linspace(0, 1, group_num + 1)).tolist()[:group_num]

        radian_list = list()
        for start_theta in start_theta_list:
            group_radian_list = get_grid_angle(step_radian, start_theta)
            radian_list += group_radian_list

        return side, radian_list

    def get_grid_crop(self, crop_width, crop_height, group_num=1):
        grid_dict = dict()
        side, radian_list = self.get_grid_angle_by_side(crop_width, self.side.outside, group_num)
        grid_dict[side] = radian_list
        side, radian_list = self.get_grid_angle_by_side(crop_width, self.side.inside, group_num)
        grid_dict[side] = radian_list

        crop_list = list()

        for key, radian_list in grid_dict.items():
            for radian in radian_list:
                cut_box, crop_box = self.get_record_from_ASP(crop_width, crop_height, radian, side=key)
                if cut_box == crop_box:
                    crop_list.append([cut_box[0], cut_box[2], cut_box[1], cut_box[3]])

        return crop_list

    def get_point_x_y_by_angle(self, angle, side=None, point_x_y=None):

        if point_x_y is not None:
            distance = self.distance_cal(point_x_y, self.circle_info['center'])
            if abs(distance - self.circle_info['radius_inside']) < abs(distance - self.circle_info['radius_outside']):
                side = self.side.inside
            else:
                side = self.side.outside
        elif side is None:
            if random.random() > 0.5:
                side = self.side.outside
            else:
                side = self.side.inside

        assert side in self.side
        radius = self.circle_info['radius_inside'] if side == self.side.inside else self.circle_info['radius_outside']
        point_y = self.circle_info['center'][1] - radius * math.sin(angle)
        point_x = self.circle_info['center'][0] + radius * math.cos(angle)

        return point_x, point_y

    def get_cut_box_by_point_x_y(self, crop_width, crop_height, point_x_y):
        crop_width, crop_height = self._crop_data_check(crop_width, crop_height)

        ori_left, ori_top, ori_right, ori_bottom = 0, 0, self.image_width-1, self.image_height-1

        crop_center_col, crop_center_row = point_x_y

        crop_left = int(crop_center_col - crop_width / 2)
        crop_right = crop_left + crop_width
        crop_top = int(crop_center_row - crop_height / 2)
        crop_bottom = crop_top + crop_height

        cut_left = max(crop_left, ori_left)
        cut_right = min(ori_right, crop_right)
        cut_top = max(ori_top, crop_top)
        cut_bottom = min(ori_bottom, crop_bottom)

        return (cut_left, cut_right, cut_top, cut_bottom), (crop_left, crop_right, crop_top, crop_bottom)

    def get_record_from_ASP(self, crop_width, crop_height, angle, side=None, ng_point_x_y=None):
        point_x_y = self.get_point_x_y_by_angle(angle, side=side, point_x_y=ng_point_x_y)
        cut_box, crop_box = self.get_cut_box_by_point_x_y(crop_width, crop_height, point_x_y)

        return cut_box, crop_box


if __name__ == "__main__":
    roi_list = GridCrop.mk_grid_roi(3000, 3000, 224, 224, 3, True)
    pass
