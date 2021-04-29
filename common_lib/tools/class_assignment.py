# -*- coding: utf-8 -*-
# @Author: Zhai Menghua
# @Date:   2020-07-27 14:58:15
# @Last Modified by:   Zhai Menghua
# @Last Modified time: 2020-07-31 14:30:00
import numpy as np

class ClassExtractor(object):
    """ Manage the assignment of class given distribution """
    def __init__(self, classname_manager, score_threshold):
        self.classname_manager = classname_manager
        self.score_threshold = score_threshold

    def extract_anno_classes(self, distribution, classnames_of_interest):
        """ return a list of groundtruth class """
        raise NotImplementedError

    def extract_pred_class(self, distribution, classnames_of_interest):
        """ return a prediction class (each image has only one prediction) """
        raise NotImplementedError


class ClassExtractorFamily(ClassExtractor):
    def extract_classes(self, distribution, classnames_of_interest):
        assert isinstance(distribution, np.ndarray), "convert {} to a np array, before you get in here".format(distribution)

        ## 1) check if OK class
        if not self.is_survival('NG', distribution):  # this is a OK distribution
            return ['OK']

        ## 2) check classnames_of_interest one by one
        survival_list = list()
        for classname in classnames_of_interest:
            if self.is_survival(classname, distribution):
                survival_list.append(classname)

        if len(survival_list) == 0:   # if the distribution is not OK, and non of the classnames_of_interest survives, return class 'Error'
            return ['Error']

        ## 3) deal with the survival_list by mode
        return survival_list

    def is_survival(self, class_name, distribution):
        """ check if the score of the current class_name and scores of all its parents are greater than the score_thresh """
        onehot_distribution = np.zeros(distribution.shape)
        onehot_distribution[self.classname_manager.get_index_from_key(class_name)] = 1.
        family_mask = np.array(self.classname_manager.extend_distribution(onehot_distribution.tolist()), dtype=np.bool)
        return all(distribution[family_mask] > self.score_threshold)

    def extract_anno_classes(self, distribution, classnames_of_interest):
        """ return a list of groundtruth class """
        survival_list = self.extract_classes(distribution, classnames_of_interest)
        return survival_list

    def extract_pred_class(self, distribution, classnames_of_interest):
        """ return a prediction class (each image has only one prediction) """
        survival_list = self.extract_classes(distribution, classnames_of_interest)

        assert len(survival_list) >= 1

        if len(survival_list) == 1:  # ['OK'], ['Error'] in this case
            return survival_list[0]
        else:
            max_score = None
            max_class = None
            for classname in survival_list:
                score = distribution[self.classname_manager.get_index_from_key(classname)]
                assert score > self.score_threshold
                if max_score is None or max_score < score:
                    max_score = score
                    max_class = classname
            assert max_class is not None
            return max_class



class ClassExtractorSingle(ClassExtractor):
    def extract_anno_classes(self, distribution, classnames_of_interest):
        """ return a list of groundtruth class """
        assert isinstance(distribution, np.ndarray), "convert {} to a np array, before you get in here".format(distribution)
        active_indices = np.where(distribution > self.score_threshold)[0]
        class_names = list()
        for index in active_indices:
            classname = self.classname_manager.get_key_from_index(index)
            if classname in classnames_of_interest:
                class_names.append(classname)
        if len(class_names) == 0:     # if no active ng class found, this class is OK class
            class_names = ['OK']
        return class_names

    def extract_pred_class(self, distribution, classnames_of_interest):
        """ return a prediction class (each image has only one prediction) """
        assert isinstance(distribution, np.ndarray), "convert {} to a np array, before you get in here".format(distribution)
        active_indices = np.where(distribution > self.score_threshold)[0]
        class_name = 'OK'
        max_score = None
        for index in active_indices:
            current_class_name = self.classname_manager.get_key_from_index(index)
            if current_class_name in classnames_of_interest:
                if max_score is None or max_score < distribution[index]:
                    max_score = distribution[index]
                    class_name = current_class_name
        return class_name


class ClassExtractorMultiClass(ClassExtractorSingle):
    def extract_anno_classes(self, distribution, classnames_of_interest):
        """ return a list of groundtruth class """
        assert isinstance(distribution, np.ndarray), "convert {} to a np array, before you get in here".format(distribution)
        class_names = list()
        for index, score in enumerate(distribution):
            assert score in [0, 1], score
            if score == 1:
                classname = self.classname_manager.get_key_from_index(index)
                if classname in classnames_of_interest:
                    class_names.append(classname)
        if len(class_names) == 0:     # if no active ng class found, this class is OK class
            class_names = ['OK']
        return class_names

    def extract_pred_class(self, distribution, classnames_of_interest):
        """
        Return a prediction class (each image has only one prediction)
        The extractor simulates the process of selecting the class of maximal score from the distribution
        Arguments:
            - distribution: distribution input with the complete classes (no 'OK')
            - classnames_of_interest: class names of interest; in our case, 
                    classnames_of_interest should be all NG classes at given level, thus SUM(score_of_interest) < 1.0
        Return:
            - class_name: either a NG class in classnames_of_interest, or 'OK'
        """
        assert isinstance(distribution, np.ndarray), "convert {} to a np array, before you get in here".format(distribution)
        class_name, max_score, ng_sum_score = None, None, 0
        for index, score in enumerate(distribution):
            current_class_name = self.classname_manager.get_key_from_index(index)
            if current_class_name in classnames_of_interest:
                ng_sum_score += score
                if max_score is None or max_score < score:
                    max_score = score
                    class_name = current_class_name
                    
        assert class_name is not None
        assert ng_sum_score < 1+1e-3, "Error({}) check if all classes come from same level: {}".format(ng_sum_score, classnames_of_interest)

        if ng_sum_score > 0.5:
            return class_name
        else:
            return 'OK'


class ClassExtractorPseudoOK(ClassExtractorSingle):
    def extract_pred_class(self, distribution, classnames_of_interest):
        """
        Return a prediction class (each image has only one prediction)
        The extractor simulates the process of selecting the class of maximal score from the distribution
        Assuming 'classnames_of_interest' does not have 'OK' class, we fake a 'OK' score with 1 - Score('NG'),
        then the prediction class is argmax(Score(cls)); in this case, there is no place for score_threshold
        Arguments:
            - distribution: distribution input with the complete classes (no 'OK')
            - classnames_of_interest: class names of interest
        Return:
            - class_name: either a NG class in classnames_of_interest, or 'OK'
        """
        assert isinstance(distribution, np.ndarray), "convert {} to a np array, before you get in here".format(distribution)
        assert 'NG' in self.classname_manager.ng_class_name_list, "there must be a 'NG' class for faking the 'OK' score"
        assert np.all(np.logical_and(distribution >= 0, distribution <= 1)), \
            "the distribution is not a valid probabilistical vector: {}".format(distribution)
        # fake the ok score
        ng_index = self.classname_manager.get_index_from_key('NG')
        pseudo_ok_score = 1. - distribution[ng_index]
        # find the class name with maximal score
        class_name = 'OK'
        max_score = pseudo_ok_score
        for index, score in enumerate(distribution):
            current_class_name = self.classname_manager.get_key_from_index(index)
            if current_class_name in classnames_of_interest:
                if max_score < score:
                    max_score = score
                    class_name = current_class_name
        return class_name


class ClassExtractorDominantNG(ClassExtractorSingle):
    def extract_pred_class(self, distribution, classnames_of_interest):
        """
        Return a prediction class (each image has only one prediction)
        A NG class shall be returned only when score(NG) > score_threshold, otherwise return 'OK'
        Arguments:
            - distribution: distribution input with the complete classes (no 'OK')
            - classnames_of_interest: class names of interest
        Return:
            - class_name: either a NG class in classnames_of_interest, or 'OK'
        """
        assert isinstance(distribution, np.ndarray), "convert {} to a np array, before you get in here".format(distribution)
        assert 'NG' in self.classname_manager.ng_class_name_list, "there must be a 'NG' class"
        ng_index = self.classname_manager.get_index_from_key('NG')
        ng_score = distribution[ng_index]
        if ng_score > self.score_threshold:
            # find the ng class name with maximal score
            class_name = None
            max_score = -np.inf
            for index, score in enumerate(distribution):
                current_class_name = self.classname_manager.get_key_from_index(index)
                if current_class_name in classnames_of_interest:
                    if max_score < score:
                        max_score = score
                        class_name = current_class_name
        else:
            class_name = 'OK'
        assert class_name is not None
        return class_name