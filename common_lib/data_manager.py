# -*- coding: utf-8 -*-
# @Author: Zhai Menghua
# @Date:   2020-07-17 10:49:13
# @Last Modified by:   Zhang Yiwei
# @Last Modified time: 2020-07-18 16:34:38
#
# A customized data manager class, derived from data-manager
#
import os
import json
import numpy as np

from .data import DataManager as DataManagerBase
from .config import JsonFormatSetting
from .tools import decode_distribution
from .tools import ClassNameManager, CnnJsonTools
from .config import cnn_json_check

from tqdm import tqdm
import numpy as np
import os
import json
import copy


class DataManager(DataManagerBase):
    def data_statistic(self, record_list):
        # 统计record
        data_statistics = dict()
        # record 总个数
        total_image_num = len(record_list)

        data = self.clone()
        data.record_list = record_list

        # 按照 缺陷类别 统计图像个数
        label2class = {x['class_id']: x['class_name'] for x in self.class_dict}

        def class_occurrence(record):
            class_names = list()
            for instance in record['instances']:
                distribution = decode_distribution(instance['distribution'])
                class_names.extend([label2class[ix] for ix, val in enumerate(distribution) if val > 0])
            return class_names

        image_statistics_by_calss_name = data.occurrence(
            lambda record: list(set(class_occurrence(record))))

        total_instance_num = data.occurrence(
            lambda record: ['total_instance_num' for _ in record["instances"]]
        )
        ng_and_ok_image_num = data.occurrence(
            lambda record: 'ng_image_num' if len(
                record["instances"]) > 0 else 'ok_image_num'
        )

        instance_statistics_by_calss_name = data.occurrence(class_occurrence)

        data_statistics['total_image_num'] = total_image_num
        data_statistics.update(ng_and_ok_image_num)
        data_statistics.update(total_instance_num)
        data_statistics['image_statistics_by_calss_name'] = image_statistics_by_calss_name
        data_statistics['instance_statistics_by_calss_name'] = instance_statistics_by_calss_name
        data_statistics['image_num_by_product'] = data.occurrence(lambda rec: rec['info']['product_id'])
        data_statistics['image_num_by_template'] = data.occurrence(lambda rec: rec['info']['template_path'])

        return data_statistics

    def save_json(self, json_file, record_callback=None, classdict_callback=None):
        """
        Save dataset to json file
        Param:
            json_file: path to the json file where you want to save your dataset
            record_callback: a callback function that processes an record and return a
                           dictionary that containes info like 'image_path' or 'instance_box' etc.
            class_dict_callback: a callback function that processes the class_dict.
        """
        json_dir = os.path.dirname(json_file)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)

        if record_callback is None:
            record_list = self.record_list
        else:
            print("Convert {} records".format(len(self)))
            record_list = list()
            for rec in tqdm(self.iterator(), total=len(self)):
                record_list.append(record_callback(rec))

        if classdict_callback is None:
            class_dict = self.class_dict
        else:
            class_dict = classdict_callback(self.class_dict)

        data_statistics = self.data_statistic(record_list)

        # save json object
        json_obj = {'version': JsonFormatSetting.VERSION, 'data_statistics': data_statistics,
                    'record': record_list, 'class_dict': class_dict}
        cnn_json_check(json_obj)
        with open(json_file, 'w', encoding='utf-8') as fid:
            json.dump(json_obj, fid, indent=4, ensure_ascii=False)

    def to_classification(self, silent=False):
        """ 
        Convert detection data to classification data in the following rules:
        1) for detection data (self), each instance has and only has one leaf node class
        2) each class (include the root and parent nodes) of the classification data is computed by Max(score(cls_i)) for i in range(#instances)
        Return:
            DataManager object for classification
        """
        data = self.clone()
        cnm = ClassNameManager(data.class_dict)
        for rec in tqdm(data, disable=silent):
            dist_cls = CnnJsonTools.make_classify_distribution_from_instance_list(rec['instances'], cnm)
            inst_cls = CnnJsonTools.instance_maker(dist_cls)
            rec['instances'] = [inst_cls]
        return data