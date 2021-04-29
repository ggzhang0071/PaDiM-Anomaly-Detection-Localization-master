# -*- coding: utf-8 -*-
# @Author: Zhai Menghua
# @Date:   2020-07-16 10:10:48
# @Last Modified by:   Zhai Menghua
# @Last Modified time: 2020-07-16 10:13:00

# -*-coding:utf-8-*-
#
# Compress and Decompress the shape object
#
import numpy as np
import copy


def encode_labelme_shape(point_list):
    """
    Encode the labelme shape (usually a list of points)
    Return a serializable object (usually a string) for json dumping
    """
    code = list()
    for point in point_list:
        assert len(point) == 2
        code.append('{:.6f}+{:.6f}'.format(point[0], point[1]))
    code = ','.join(code)
    return code


def decode_labelme_shape(encoded_shape):
    """
    Decode the cnn json shape (usually encoded from labelme format)
    Return a list of points that are used in labelme
    """
    assert isinstance(encoded_shape, str)
    points = encoded_shape.split(',')
    shape = list()
    for point in points:
        x, y = point.split('+')
        shape.append([float(x), float(y)])
    return shape


def encode_distribution(distribution):
    """
    Encode the distribution (usually a list of float)
    Return a serializable object (usually a string) for json dumping
    """
    code = list()
    for data in distribution:
        if None:
            pass
        elif isinstance(data, float) or isinstance(data, np.float32):
            code.append('{:0.6e}'.format(data))
        elif isinstance(data, int):
            code.append(str(data))
        else:
            raise RuntimeError
    code = ','.join(code)
    return code


def decode_distribution(encoded_distribution):
    """
    Decode the cnn json distribution (usually encoded data)
    Return a list of float
    """
    if isinstance(encoded_distribution, list):
        return encoded_distribution
    assert isinstance(encoded_distribution, str)
    distribution_str_list = encoded_distribution.split(',')
    distribution = list()
    for data_str in distribution_str_list:
        data = float(data_str)
        distribution.append(data)
    return distribution


def extend_distribution(input_distribution, class_name_manager):
    '''
    extend one-hot distribution to its ancestors. all of the ancestors will be
    set to 1
    NOTE: This function is DEPRECIATED, use class_name_manager.extend_distribution() instead!!!
    args:
        input_distribution: a list of distribution, shuld be onehot label and the sum must be 1, could be a str or list
        class_name_manager: instance of ClassNameManager
    return
        current_distribution: exetend distribution(list consists of float nums)
    '''
    return class_name_manager.extend_distribution(input_distribution)


def is_onehot_distribution(input_distribution):
    '''
    Determine whether the distribution is onehot
    args
        input_distribution: input_distribution, could be a str or list
    return
        bool: onehot-True or not-onehot-False
    '''
    current_distribution = decode_distribution(input_distribution)
    assert isinstance(current_distribution, list), "the decoded_distribution should be a list"
    array_distribution = np.array(current_distribution)
    if np.sum(array_distribution) == 1 and is_bool_distribution(current_distribution):
        return True
    else:
        return False


def is_bool_distribution(input_distribution):
    '''
    Determine whether the distribution is bool
    args
        input_distribution: input_distribution, could be a str or list
    return
        bool: bool-True or not-bool-False
    '''
    current_distribution = decode_distribution(input_distribution)
    assert isinstance(current_distribution, list), "the decoded_distribution should be a list"
    if set(current_distribution).issubset({0.0, 1.0}):
        return True
    else:
        return False


if __name__ == "__main__":

    # encode_labelme_shape demo
    shape = [
        [
            755.6206896551724,
            116.90804597701151
        ],
        [
            700.448275862069,
            245.64367816091954
        ],
        [
            832.6321839080459,
            273.2298850574713
        ],
        [
            894.7011494252873,
            181.27586206896552
        ],
        [
            842.9770114942529,
            103.11494252873564
        ]
    ]

    code = encode_labelme_shape(shape)
    shape_rec = decode_labelme_shape(code)

    print(shape_rec)
    print(code)

    # encode_labelme_shape demo
    import random
    distribution = [random.random() for i in range(30)]
    print(distribution)
    encoded_distribution = encode_distribution(distribution)
    print(encoded_distribution)
    distribution = decode_distribution(encoded_distribution)
    print(distribution)

    distribution = [random.randint(0, 1) for i in range(30)]
    print(distribution)
    encoded_distribution = encode_distribution(distribution)
    print(encoded_distribution)
    distribution = decode_distribution(encoded_distribution)
    print(distribution)
