from data_manager import DataManager
import os, copy

def groupID_callback(x):
    """
    image_name = 1_1-1256_1264_2461_2455-12003110450161(_1).jpg
    big_image_name = 12003110450161
    """
    img_path = x['info']['image_path']
    big_img_name = img_path.split('.')[-2]
    big_img_name = big_img_name.split('-')[-1].split('_')[0]
    return big_img_name

def record_callback(x):
    info = {
        'image_path': os.path.basename(x['info']['image_path']),
        'parent_image_path': groupID_callback(x)
    }
    pred_inst = list()
    for inst in x['pred_inst']:
        inst_new = copy.copy(inst)
        bb = inst_new.pop('bound_box')
        X,Y,W,H = bb['x_left'], bb['y_top'], bb['width'], bb['height']
        inst_new['instance_bbox'] = [X,Y,X+W,Y+H]
        pred_inst.append(inst_new)
    return {'info':info, 'pred_inst':pred_inst}

def classdict_callback(class_dict):
    class_dict_new = list()
    for class_item in class_dict:
        class_item_new = copy.copy(class_item)
        class_item_new['class_id'] = int(class_item['class_name'])
        class_dict_new.append(class_item_new)
    return class_dict_new

def key_callback(x):
    """ extract class_names from all instances detected on a image """
    keys = list()
    for inst in x['pred_inst']:
        keys.append(inst['class_name'])
    return keys

def main():
    data = DataManager.from_json('assets/toy.json')
    data_train, data = data.split(num_or_ratio=.7, groupID_callback=groupID_callback)
    data_val, data_test = data.split(num_or_ratio=.333, groupID_callback=groupID_callback)

    # save to jsons
    data_train.save_json(
        'tmp/train.json', 
        record_callback=record_callback, classdict_callback=classdict_callback   # save data_train in new format json
    )
    data_test.save_json('tmp/test.json')
    data_val.save_json('tmp/val.json')

    print("Occurrence of classes:")
    occurrence = data.occurrence(key_callback=key_callback)
    print(occurrence)

if __name__ == '__main__':
    main()