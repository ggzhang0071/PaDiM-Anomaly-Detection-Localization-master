# -*- coding: utf-8 -*-
# @Author: Zhai Menghua
# @Date:   2020-07-17 11:01:04
# @Last Modified by:   Zhai Menghua
# @Last Modified time: 2020-07-21 20:59:00
#
# Data Manager Class
#
import copy, random, os, json
from tqdm import tqdm

class DataManager(object):
    def __init__(self, record_list, class_dict):
        assert isinstance(record_list, (list, tuple))
        self.record_list = record_list
        self.class_dict = class_dict

    def filter(self, condition_callback, verbose=False):
        """
        Filter data by the given condition
        Param:
            condition_callback: a callback function that returns a boolean value according to attributes
            verbose: if set True, then print information about the operation
        Return:
            a DataManager instance, whose record_list satisfies the given condition
        """
        try:
            if verbose:
                record_list = [copy.deepcopy(x) for x in tqdm(self.iterator(), total=len(self)) if condition_callback(x)]
            else:
                record_list = [copy.deepcopy(x) for x in self.iterator() if condition_callback(x)]
        except RuntimeError as e:
            print("condition_callback Error:", e)
        data = type(self)(record_list, copy.copy(self.class_dict))
        if verbose:
            print("{} out of {} images are found meeting the given condition".format(len(record_list), len(self)))
        return data

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        return self.record_list[idx]

    def clone(self):
        return copy.deepcopy(self)

    def iterator(self):
        return iter(self.record_list)

    def split(self, num_or_ratio, groupID_callback=None, random_seed=123):
        """
        Split data by the given condition
        Param:
            num_or_ratio: number (int) or ratio (float) of the first half of the splits (by groupID)
            groupID_callback: a callback function that returns a hashable object (groupID); if unset, split the dataset itemwise
        Return:
            two DataManager instances, the groupID computed from the record_list 
            of one differs from the values computed from the record_list of the other
        """
        # 1) put data in group (by their groupID)
        groups = dict()
        for ix, record in enumerate(self.iterator()):
            if groupID_callback == None:
                group_id = ix
            else:
                try:
                    group_id = groupID_callback(record)
                except RuntimeError as e:
                    print("groupID_callback Error:", e)
            if group_id not in groups:
                groups[group_id] = [record]
            else:
                groups[group_id].append(record)
        num_groups = len(groups)

        # 2) split data by groups
        if None: pass
        elif isinstance(num_or_ratio, float):
            assert 0 <= num_or_ratio <= 1
            num_firsthalf = int(num_groups * num_or_ratio)
        elif isinstance(num_or_ratio, int):
            num_firsthalf = num_or_ratio

        assert num_firsthalf <= num_groups, "There're less than {} groups to split: {} < {}".format(num_firsthalf, num_groups, num_firsthalf)

        # split data
        groups = [groups[key] for key in sorted(groups.keys())]  # sort groups by key for repeatibility
        random.seed(random_seed)
        random.shuffle(groups)
        firsthalf, lasthalf = groups[:num_firsthalf], groups[num_firsthalf:]
        record_list1 = list()
        for grp in firsthalf:
            record_list1.extend([copy.deepcopy(x) for x in grp])
        record_list2 = list()
        for grp in lasthalf:
            record_list2.extend([copy.deepcopy(x) for x in grp])

        data1 = type(self)(record_list1, copy.copy(self.class_dict))
        data2 = type(self)(record_list2, copy.copy(self.class_dict))

        return data1, data2

    def merge(self, another_dataset):
        """
        Fuse two datasets
        Param:
            another_dataset: a DataManager instance
        Return:
            a DataManager instance that fused self and another_dataset
        """
        assert isinstance(another_dataset, type(self))
        assert self.class_dict == another_dataset.class_dict
        record_list_merged = [copy.deepcopy(x) for x in self.record_list+another_dataset.record_list]
        data_merged = type(self)(record_list_merged, copy.copy(self.class_dict))
        return data_merged

    def dump(self):
        """ Print all info of the dataset """
        import pprint
        for ix, record in enumerate(self.iterator()):
            print("\n[{}/{}] sample >>".format(ix+1, len(self)))
            pprint.pprint(record)
        print("\nClass Dict:")
        print(self.class_dict)

    def extract_info(self, info_callback):
        """
        Extract infomation extracted by the info_callback function
        Param:
            info_callback: callback function that extracts info from record
        Return:
            a list of info items extracted by info_callback
        """
        return list(map(info_callback, self.iterator()))

    def occurrence(self, key_callback):
        """
        Count the occurrence of record
        Param:
            key_callback: a callback function that returns a hashable object (ie. groupID) or a list of hashable objects
        Return:
            a dictionary: {key_callback(record1): occurrence1, ...}
        """
        occurrence = dict()
        for record in self.iterator():
            try:
                key = key_callback(record)
            except RuntimeError as e:
                print("key_callback Error:", e)
            # if the extract key is a list (or tuple)
            if isinstance(key, list) or isinstance(key, tuple):
                keys = key
            else:
                keys = [key]
            for key in keys:
                if not key in occurrence:
                    occurrence[key]  = 1
                else:
                    occurrence[key] += 1
        return occurrence

    def unique(self, key_callback, verbose=False):
        """
        Remove duplicated record by its key (the removal choices are made randomly)
        Param:
            key_callback: a callback function that returns a hashable object (ie. groupID) or a list of hashable objects
            verbose: if set True, then print information about the operation
        Return:
            a new DataManager object, whose records have unique key returned by key_callback
        """
        class_dict = copy.deepcopy(self.class_dict)
        record_list = list()
        visited_key = set()
        for record in self.iterator():
            try:
                key = key_callback(record)
            except RuntimeError as e:
                print("key_callback Error:", e)
            if key not in visited_key:
                record_list.append(copy.deepcopy(record))
                visited_key.add(key)
        if verbose:
            print("{} out of {} duplicated records are found".format(len(self)-len(record_list), len(self)))
        return type(self)(record_list=record_list, class_dict=class_dict)

    def intersection(self, another_dataset, key_callback, verbose=False):
        """
        Find the intersection set of two datasets, if two records share the same key, they are deemed as members of the intersection set
        Param:
            another_dataset: a DataManager instance
            key_callback: a callback function that returns a hashable object (ie. image uuid or image hash code)
            verbose: if set True, then print information about the operation
        Return:
            a DataManager instance that represents for the intersection of two datasets
        """
        assert isinstance(another_dataset, type(self))
        assert self.class_dict == another_dataset.class_dict
        class_dict = copy.deepcopy(self.class_dict)
        intersection_keys = set(another_dataset.extract_info(info_callback=key_callback))
        record_list = list()
        for record in self.iterator():
            try:
                key = key_callback(record)
            except RuntimeError as e:
                print("key_callback Error:", e)
            if key in intersection_keys:
                record_list.append(copy.deepcopy(record))
        if verbose:
            print("{} out of {} overlapping records are found".format(len(record_list), len(self)))
        return type(self)(record_list=record_list, class_dict=class_dict)
        
    def difference(self, another_dataset, key_callback, verbose=False):
        """
        Find the difference set of two datasets, whose members exist in self but not in another_dataset
        Param:
            another_dataset: a DataManager instance
            key_callback: a callback function that returns a hashable object (ie. image uuid or image hash code)
            verbose: if set True, then print information about the operation
        Return:
            a DataManager instance that represents for the difference set of two datasets
        """
        assert isinstance(another_dataset, type(self))
        assert self.class_dict == another_dataset.class_dict
        class_dict = copy.deepcopy(self.class_dict)
        intersection_keys = set(another_dataset.extract_info(info_callback=key_callback))
        record_list = list()
        for record in self.iterator():
            try:
                key = key_callback(record)
            except RuntimeError as e:
                print("key_callback Error:", e)
            if key not in intersection_keys:
                record_list.append(copy.deepcopy(record))
        if verbose:
            print("{} out of {} different records are found".format(len(record_list), len(self)))
        return type(self)(record_list=record_list, class_dict=class_dict)

    def save_json(self, json_file, record_callback=None, classdict_callback=None, extra_fields=None):
        """
        Save dataset to json file
        Param:
            json_file: path to the json file where you want to save your dataset
            record_callback: a callback function that processes an record and return a
                           dictionary that containes info like 'image_path' or 'instance_box' etc.
            class_dict_callback: a callback function that processes the class_dict.
            extra_fields: a dictionary that would be saved in json at the same level with 'class_dict' and 'record'
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

        json_object = {'class_dict': class_dict, 'record': record_list}
        if extra_fields is not None:
            assert isinstance(extra_fields, dict), "'extra_fields' must be a dictionary"
            json_object.update(extra_fields)

        # save json object
        with open(json_file, 'w') as fid:
            json.dump(
                json_object,
                fid,
                indent = 4
            )

    @classmethod
    def from_json(cls, json_file):
        """
        Load dataset from json file
        Param:
            json_file: path to the json file where you want to read. Its format has to satisfy: {'class_dict': ..., 'record': ...}
        Return:
            a DataManager instance
        """
        assert os.path.exists(json_file), "{} doesn't exist".format(json_file)
        with open(json_file) as fid:
            obj = json.load(fid)

        assert 'class_dict' in obj and 'record' in obj, "invald json file to load: {}".format(json_file)
        class_dict = obj['class_dict']
        record_list = list()
        for rec in obj['record']:
            record_list.append(rec)
        dataset = cls(record_list=record_list, class_dict=class_dict)

        return dataset

    def batch(self, batch_size, random_seed=123):
        """
        Divide the dataset into a series of batches, randomly
        Param:
            batch_size: #records this method will return per iteration
        Return:
            an iterator of DataManager objects
        """
        assert isinstance(batch_size, int), "batch_size should be a natural number"
        assert 0 < batch_size <= len(self)
        ds = self.clone()
        record_list = ds.record_list
        random.seed(random_seed)
        random.shuffle(record_list)
        for idx in range(0, len(record_list), batch_size):
            records = record_list[idx:idx+batch_size]
            yield type(self)(record_list=records, class_dict=ds.class_dict)

    def chunk(self, chunk_num, random_seed=123):
        """
        Chunk the dataset (divide it into a given number of chunks), randomly
        Param:
            chunk_num: #chunks this method will return
        Return:
            an iterator of DataManager objects
        """
        assert isinstance(chunk_num, int), "chunk_num should be a natural number"
        assert 0 < chunk_num <= len(self)
        batch_size = len(self) // chunk_num
        if len(self) % chunk_num > 0:
            batch_size += 1
        return self.batch(batch_size, random_seed)
