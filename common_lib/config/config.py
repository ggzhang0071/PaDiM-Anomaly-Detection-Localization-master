# -*- coding: utf-8 -*-
# @Author: Zhai Menghua
# @Date:   2020-07-22 15:52:10
# @Last Modified by:   Zhai Menghua
# @Last Modified time: 2020-07-23 10:17:09

from collections import OrderedDict

class JsonFormatSettingClass(object):
    VERSION = '2.0.0' # add template alignment

    CNN_JSON_STRUCTURE = {
        "version": str,
        "class_dict": [
            {
                "class_name": str,
                "class_id": int,
                "parent": None,
            },
        ],
        "record": [
            {
                "info": {
                    "uuid": str,
                    "image_path": str,
                    "template_path": None,
                    'template_offset_xy': [int, int],
                    'template_scale': float,
                    "roi": [int],
                    "side": str,
                    "product_id": str,
                    "lot_id": str,
                    "width": int,
                    "height": int
                },
                "instances": [
                    {
                        "uuid": str,
                        "distribution": str,
                        "shape_type": str,
                        "points": str
                    }
                ]
            },
        ]
    }

JsonFormatSetting = JsonFormatSettingClass()
