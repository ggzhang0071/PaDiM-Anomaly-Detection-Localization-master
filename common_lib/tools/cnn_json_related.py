# -*- coding: utf-8 -*-
# @Author: Zhai Menghua
# @Date:   2020-07-16 10:10:07
# @Last Modified by:   Zhang Yiwei
# A unified tool for updating CNN JSON with network prediction results
import copy
import sys

import numpy as np
from tqdm import tqdm
from enum import Enum
from .id_related import create_uuid

from .coding_related import encode_distribution
from .coding_related import encode_labelme_shape
from .coding_related import decode_distribution
from .coding_related import extend_distribution
from .coding_related import is_onehot_distribution


class CnnJsonTools:
    '''
    class to update cnn json
    '''
    @classmethod
    def instance_maker(cls, distribution, points=[], shape_type='polygon', force=False):
        '''
        make a cnn_json instance
        args
            :distribution:  a list; there would be just one 1 in the list,
                            while others are 0, which presents a leaf node of class dict tree
            :shape_type:    shape of the type, one of ['linestrip', 'polygon', 'rectangle']
            :points:        a list of points
        return
            :instance: a instance dict
        '''
        # assert sum(distribution) == 1
        assert shape_type in ['linestrip', 'polygon', 'rectangle']

        instance = dict()
        instance['uuid'] = create_uuid()
        if isinstance(distribution, list):
            instance['distribution'] = encode_distribution(distribution)
        elif isinstance(distribution, str):
            instance['distribution'] = distribution
        else:
            if force:
                instance['distribution'] = distribution
            else:
                raise RuntimeError("Unrecognizable distribution type: {}, if you really want to save this type of distribution, set force True".format(type(distribution)))
        instance['shape_type'] = shape_type
        if isinstance(points, list):
            instance['points'] = encode_labelme_shape(points)
        elif isinstance(points, str):
            instance['points'] = points
        else:
            raise RuntimeError("Unrecognizable distribution type: {}".format(points))
        return instance

    @classmethod
    def make_classify_distribution_from_instance_list(cls, instance_list, class_name_manager):
        '''
        return a classify distribution of the image throuth the instance_list
        args
            instance_list: instance list of a record
            class_name_manager: instance of ClassNameManager
        return
            classify_distribution: distribution of the image (a list) and all of the parent nodes would be set to 1
        '''
        assert isinstance(instance_list, list)
        ng_class_num = len(class_name_manager.ng_class_name_list)
        distribution_matrix = list()
        distribution_matrix.append(np.zeros((ng_class_num), dtype='uint8').tolist())
        for instance in instance_list:
            if is_onehot_distribution(instance['distribution']):
                sub_distribution = np.array(extend_distribution(instance['distribution'], class_name_manager))
            else:
                sub_distribution = decode_distribution(instance['distribution'])
            distribution_matrix.append(sub_distribution)
        distribution_matrix = np.array(distribution_matrix)
        classify_distribution = distribution_matrix.max(axis=0)
        return classify_distribution.tolist()


class ClassNameManager:
    '''
    manage info of class_dict loaded from a cnn_json
    '''

    def __init__(self, class_dict):
        self.original_class_dict = class_dict
        self.class_dict_check(class_dict)
        self.class_name_tree_dict = self.make_class_dict_tree(class_dict)
        self.classname_hierarchy = self.generate_classname_hierarchy()
        self.ng_class_name_list, self.whole_class_name_list = self.get_ng_class_name_list(class_dict)
        self.parent_list = self.get_parent_list(self.class_name_tree_dict)
        self.children_list = self.get_children_list(self.class_name_tree_dict)
        self.ng_class_num = len(self.ng_class_name_list)
        self.total_class_num = len(self.whole_class_name_list)

        # generate class2label, label2class by breadth-first propagate the self.class_name_tree_dict
        # such that the label2class are repeatable for the same dataset (class_dict)
        self.label2class = {x['class_id']:x['class_name'] for x in class_dict}
        self.class2label = {x['class_name']:x['class_id'] for x in class_dict}
        
        self.label2class_children = dict()
        self.class2label_children = dict()
        label = 0
        for x in class_dict:
            class_name = x['class_name']
            label_parent = x['class_id']
            if class_name in self.children_list:
                self.label2class_children[label] = class_name
                self.class2label_children[class_name] = label
                label += 1


    def class_dict_check(self, class_dict):
        """
        check the validity of input class dict
        """
        assert isinstance(class_dict, list)
        num = len(class_dict)
        index_list = [index for index in range(num)]

        # class id check
        for item in class_dict:
            index = item['class_id']
            index_list.remove(index)
        assert index_list == [], "class dict id error {}".format(class_dict)

        # class name and parent check
        name_list = list()
        for item in class_dict:
            class_name = item['class_name']
            assert 'parent' in item, "a class dict must contains a 'parent' {}".format(item)
            if class_name not in name_list:
                assert isinstance(class_name, str) and len(class_name) > 0, "Illegal class name {}".format(class_name)
                name_list.append(class_name)
            else:
                raise ValueError("Duplicated class name {}".format(class_name))

    def __eq__(self, another_class_name_manager):
        """
        overloading '==' operator 
        """
        if not self.original_class_dict == another_class_name_manager.original_class_dict:
            return False
        if not self.class_name_tree_dict == another_class_name_manager.class_name_tree_dict:
            return False
        if not self.ng_class_name_list == another_class_name_manager.ng_class_name_list:
            return False
        if not self.parent_list == another_class_name_manager.parent_list:
            return False
        if not self.children_list == another_class_name_manager.children_list:
            return False
        if not self.ng_class_num == another_class_name_manager.ng_class_num:
            return False
        if not self.total_class_num == another_class_name_manager.total_class_num:
            return False
        return True

    def build_distribution_from_index(self, index, extend=False):
        """
        build a one-hot distribution form index
        """
        assert isinstance(index, int)
        assert -1 <= index < self.ng_class_num
        distribution = [0 for i in range(self.ng_class_num)]
        if index != -1:
            distribution[index] = 1
        if extend:
            return self.extend_distribution(distribution)
        else:
            return distribution

    def build_distribution_from_key(self, key, extend=False):
        """
        build a one-hot distribution form a key
        """
        if key == 'OK':
            distribution = self.build_distribution_from_index(-1)
        else:
            assert key in self.ng_class_name_list, "class name '{}' not found in class dict".format(key)
            index = self.get_index_from_key(key)
            distribution = self.build_distribution_from_index(index)
        if extend:
            return self.extend_distribution(distribution)
        else:
            return distribution

    def get_key_from_distribution(self, distribution):
        """
        get key list from distribution
        data > 0 will be chosen
        """
        decoded_distribution = decode_distribution(distribution)
        assert len(decoded_distribution) == self.ng_class_num
        chosen_distribution = np.where(np.array(decoded_distribution) > 0)[0].tolist()
        return self.get_key_from_index(chosen_distribution)

    def make_key_value_dict_form_distribution(self, distribution):
        """
        build a dict
        keys are the strs in ng_class_name_list
        values comes from the corresponding data in distribution
        """
        decoded_distribution = decode_distribution(distribution)
        assert len(decoded_distribution) == self.ng_class_num
        key_value_dict = dict()
        for key, value in zip(self.ng_class_name_list, decoded_distribution):
            key_value_dict[key] = value
        return key_value_dict

    def get_key_from_index(self, index):
        """
        get key list from index
        the index input can be a int or a list
        """
        def get_key(index):
            assert 0 <= index <= len(self.ng_class_name_list)
            return self.ng_class_name_list[index]
        if isinstance(index, (int, np.int64)):
            return get_key(index)
        elif isinstance(index, list):
            key_list = list()
            for sub_index in index:
                key_list.append(get_key(sub_index))
            return key_list
        else:
            raise RuntimeError("index type must be int or list")

    def get_index_from_key(self, key):
        """
        get index list from key
        the key input can be a str or a list
        """
        def get_index(key):
            assert key in self.ng_class_name_list
            return self.class_name_tree_dict[key]['class_id']
        if isinstance(key, str):
            return get_index(key)
        elif isinstance(key, list):
            index_list = list()
            for sub_key in key:
                index_list.append(get_index(sub_key))
            return index_list
        else:
            raise RuntimeError("key type must be str or list")

    def make_class_dict_tree(self, class_dict):
        """
        build class dict tree from class_dict input
        """
        class_name_tree_dict = dict()
        for item in class_dict:
            temp_dict = item.copy()
            temp_dict['children'] = list()
            class_name_tree_dict[temp_dict['class_name']] = temp_dict

        for key, value in class_name_tree_dict.items():
            if value['parent']:
                class_name_tree_dict[value['parent']]['children'].append(key)

        return class_name_tree_dict

    def get_ng_class_name_list(self, class_dict):
        """
        get ng class name list from class_dict
        the class_name_list is ordered by class_id of class_dict
        """
        ng_class_name_list = ["" for _ in class_dict]
        for item in class_dict:
            ng_class_name_list[item['class_id']] = item['class_name']
        whole_class_name_list = ['OK'] + ng_class_name_list
        return ng_class_name_list, whole_class_name_list

    def get_parent_list(self, class_name_tree_dict):
        """
        get the node list who has children
        """
        parent_list = list()
        for key, value in class_name_tree_dict.items():
            if len(value['children']) > 0:
                parent_list.append(key)
        return parent_list

    def get_children_list(self, class_name_tree_dict):
        """
        get the node list who has no children
        """
        children_list = list()
        for key, value in class_name_tree_dict.items():
            if len(value['children']) == 0:
                children_list.append(key)
        return children_list

    def generate_classname_hierarchy(self):
        """
        return a list of list of ng class names, each list item contains all class names from the same level of hierarchy
        for example:  ['Foreign_object', 'Bump_class', 'Wire_class', 'PI_class', 'IQC_class', 'Others'] belongs to the second level of class group
        """
        assert hasattr(self, 'class_name_tree_dict')
        # find root class(ies), whose parent is None
        root_class_list = [class_name for class_name, class_info in self.class_name_tree_dict.items() if class_info['parent'] is None]
        class_groups = [root_class_list]
        reach_end = False
        while not reach_end:
            parent_names = class_groups[-1]
            children_names = list()
            reach_end = True
            for class_name in parent_names:
                class_info = self.class_name_tree_dict[class_name]
                children = class_info['children']
                if len(children) == 0:  # if no children for this parent, then add it to the children generation
                    children_names.append(class_name)
                else:
                    children_names.extend(children)
                    reach_end = False  # if current generation exists, then keep exploring
            class_groups.append(children_names)
        class_groups.pop()  # the last group is a sentinal group (invalid), so pop it
        return class_groups

    def extend_distribution(self, input_distribution):
        '''
        extend one-hot distribution to its ancestors. all of the ancestors will be
        set to 1
        args:
            input_distribution: a list of distribution, shuld be onehot label and the sum must be 1, could be a str or list
        return
            current_distribution: exetend distribution(list consists of float nums)
        '''
        current_distribution = copy.deepcopy(input_distribution)
        current_distribution = decode_distribution(current_distribution)
        assert isinstance(current_distribution, list), "the decoded_distribution should be a list"
        assert len(current_distribution) == len(self.class_name_tree_dict)
        assert np.sum(current_distribution) == 1., current_distribution
        max_index = np.argmax(current_distribution)
        class_name = self.get_key_from_index(max_index)
        parent = self.class_name_tree_dict[class_name]['parent']
        lock_mark = 0
        while parent is not None:
            new_index = self.get_index_from_key(parent)
            current_distribution[new_index] = 1
            class_name = self.get_key_from_index(new_index)
            parent = self.class_name_tree_dict[class_name]['parent']
            lock_mark += 1
            if lock_mark > 1000:
                raise RuntimeError("endless loop")
        return current_distribution

    def is_valid_gt_distribution(self, gt_distribution):
        """
        Check whether the truth distribution consists of 0 and 1 and whether it conforms to the tree structure
        args:
            gt_distribution: a list of distribution, shuld be onehot label and the sum must be 1, could be a str or list
        return:
            if a class node is active (score=1) while having a inactive parent (score=0), then return False, otherwise return True
            all-0 distribution is also valid, so return True
        """
        current_distribution = copy.deepcopy(gt_distribution)
        current_distribution = decode_distribution(current_distribution)
        assert isinstance(current_distribution, (list, np.ndarray))

        if len(current_distribution) != self.ng_class_num:
            return False
        if set(current_distribution) | {0, 1} != {0, 1}:
            return False

        label_list = [idx for idx, label in enumerate(current_distribution) if label == 1]
        id2name = {v['class_id']: v['class_name'] for k, v in self.class_name_tree_dict.items()}
        name2id = {v['class_name']: v['class_id'] for k, v in self.class_name_tree_dict.items()}
        for label_index in label_list:
            label_dict = self.class_name_tree_dict[id2name[label_index]]
            parent = label_dict['parent']

            assert isinstance(parent, (type(None), str))

            if parent is None:
                continue

            parent_index = name2id[parent]
            if parent_index not in label_list:
                return False
        return True

    def dist_det2cls(self, distributions, label2class):
        """
        Turn the leaf-class detection distribution to the full-class classification distribution
        Arguments:
            distributions: N x #leaf-class nd array, represents N leaf-class distribution (detection output)
            label2class: a dictionary that records class names in position of distributions: {0: 'Bump-blur', 1: 'RDL-defect', ...}
        Return:
            N x #full-class nd array, represents N full-class distribution (classification output)
        """
        assert isinstance(distributions, np.ndarray)
        leaf_class_num = len(self.children_list)
        full_class_num = self.ng_class_num
        label2class_leaf = label2class
        class2label_full = self.class2label
        assert distributions.shape[-1] == leaf_class_num and len(label2class) == leaf_class_num

        leaf2full_ids = [class2label_full[label2class_leaf[label]] for label in range(leaf_class_num)]
        distributions_full = np.zeros([distributions.shape[0], len(self.class2label)])
        for idx, leaf2full_idx in enumerate(leaf2full_ids):
            # fill leaf class score
            scores = distributions[:,idx]
            distributions_full[:,leaf2full_idx] = scores
            # update parent class scores
            parent_name = self.class_name_tree_dict[label2class_leaf[idx]]['parent']
            while parent_name is not None:
                parent_id = self.get_index_from_key(parent_name)
                distributions_full[:,parent_id] += scores
                parent_name = self.class_name_tree_dict[parent_name]['parent']
        return distributions_full


class ClassMapper(object):
    """
    build a map realationship between two classname managers and check the validity
    """
    SubSetKeyList = ['A', 'B', 'C', 'D']

    def __init__(self, source_classname_manager, target_classname_manager, mapper_dict):
        self.source_classname_manager = source_classname_manager
        self.target_classname_manager = target_classname_manager
        self.mapper_dict = mapper_dict
        self.mapper_check(self.source_classname_manager, self.target_classname_manager, self.mapper_dict)

    class OutputFormat(Enum):
        # output format enum
        LIST = 'list'
        STR = 'str'

    def distribution_mapping(self, input_distribution, res_format=None):
        """
        map input_distribution from source_classname_manager to target_classname_manager through mapper_dict
        args:
            input_distribution: distribution input, can be a str or list
            res_format: output result distribution format, if is None, it will be set to the same type with input distribution
        returns:
            mapped_distribution: mapped distribution in type of res_format
        """
        # set res_format
        if res_format is None:
            if isinstance(input_distribution, str):
                res_format = self.OutputFormat.STR
            elif isinstance(input_distribution, list):
                res_format = self.OutputFormat.LIST
            else:
                raise RuntimeError("input_distribution error typt {}".format(input_distribution))

        # input distribution check
        decoded_distribution = decode_distribution(input_distribution)
        assert isinstance(decoded_distribution, list)
        assert len(decoded_distribution) == self.source_classname_manager.ng_class_num

        # mapping distribution
        mapped_distribution = [sys.maxsize] * self.target_classname_manager.ng_class_num

        def cal_mapped_value(dict, source_classname_manager, decoded_distribution):
            #                A      D
            #                 \     | \
            #                  \    |  \
            #                   \   B   C
            #                    \  |
            #                    b\ |a
            #                       E
            """
            calculate mapped value E through formula :
            a = D -[D - max(B)]·max(C)
            if a list is mepty, we make its max value 0
            if D is None, we make D 0
            b = max(A)
            E = max(a, b)
            """
            assert list(dict.keys()) == self.SubSetKeyList

            def get_value_list_from_key_list(key_list):
                value_list = list()
                if isinstance(key_list, list):
                    for key in key_list:
                        index = source_classname_manager.get_index_from_key(key)
                        value_list.append(decoded_distribution[index])
                elif isinstance(key_list, str):
                    index = source_classname_manager.get_index_from_key(key_list)
                    value_list.append(decoded_distribution[index])
                elif key_list is None:
                    return 0
                if len(value_list) == 0:
                    return 0
                else:
                    return max(value_list)

            pred_A = get_value_list_from_key_list(dict['A'])
            pred_B = get_value_list_from_key_list(dict['B'])
            pred_C = get_value_list_from_key_list(dict['C'])
            pred_D = get_value_list_from_key_list(dict['D'])

            mapped_value = max(pred_A, pred_D - (pred_D - pred_B) * pred_C)
            return mapped_value

        # calculate mapped_distribution
        for key, map_dict in self.mapper_dict.items():
            mapped_distribution_index = self.target_classname_manager.get_index_from_key(key)
            mapped_value = cal_mapped_value(map_dict, self.source_classname_manager, decoded_distribution)
            mapped_distribution[mapped_distribution_index] = mapped_value

        assert isinstance(mapped_distribution, list)
        assert sys.maxsize not in mapped_distribution
        assert len(mapped_distribution) == self.target_classname_manager.ng_class_num

        if res_format == self.OutputFormat.STR:
            return encode_distribution(mapped_distribution)
        elif res_format == self.OutputFormat.LIST:
            return mapped_distribution

    @classmethod
    def mapper_check(cls, source_classname_manager, target_classname_manager, mapper_dict):
        """
        check the validity of the input parameters
        """
        assert list(mapper_dict) == target_classname_manager.ng_class_name_list

        def get_source_node_list_from_mapper(mapper_dict):
            source_node_list = list()
            set_name_list = cls.SubSetKeyList
            for _, value in mapper_dict.items():
                assert isinstance(value, dict)
                for sub_key, sub_value in value.items():
                    assert sub_key in set_name_list
                    if isinstance(sub_value, list):
                        source_node_list.extend(sub_value)
                    elif isinstance(sub_value, str):
                        source_node_list.append(sub_value)
                    else:
                        assert sub_value is None, "{} must be list/str or None".format(sub_value)
            return source_node_list

        source_node_list = get_source_node_list_from_mapper(mapper_dict)
        for source_node in source_node_list:
            assert source_node in source_classname_manager.ng_class_name_list
        assert set(source_node_list) == set(source_classname_manager.ng_class_name_list)
        return True

    @classmethod
    def from_dict(cls, sorce_classdict, target_classdict, mapper_dict):
        """
        build instance of ClassMapper through class_dicts and mapper_dict
        """
        source_classname_manager = ClassNameManager(sorce_classdict)
        target_classname_manager = ClassNameManager(target_classdict)
        return cls(source_classname_manager, target_classname_manager, mapper_dict)
