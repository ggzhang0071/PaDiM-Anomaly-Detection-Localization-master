#
# common tools about json
#

from .config import JsonFormatSetting


def check_dict(my_record, sta_record, key=""):
    """
    check validity of the  input dictionaries Recursively
    args
    :my_record: the dict to be checked
    :sta_record: standard structure of the dict
    return
    : if sth is wrong, an assert would tell you what happened
    : if everything is ok, a True will be returned
    """
    if isinstance(sta_record, dict):
        assert isinstance(my_record, dict), '{} shoule be a dict'.format(my_record)
        for item in tuple(sta_record):
            assert item in tuple(my_record), "{0} not found in json object".format(item)
            check_dict(my_record[item], sta_record[item], key=item)

    elif isinstance(sta_record, list):
        for item in my_record:
            check_dict(item, sta_record[0])

    elif sta_record is None:
        pass

    else:
        assert isinstance(my_record, sta_record), "{} - {} type error, it is supposed to be a {}".format(key, my_record, sta_record)

    return True


def record_check(record):
    """
    record dict check
    --- a dictionary is required as the input ---
    """
    assert isinstance(
        record, dict), 'record should be dict, while the input is {}'.format(type(record))
    cnn_json_struct = JsonFormatSetting.CNN_JSON_STRUCTURE
    record_struct = cnn_json_struct["record"][0]

    return check_dict(record, record_struct)


def cnn_json_check(cnn_json_dict):
    """
    cnnjson dict check
    --- a dictionary is required as the input ---
    """
    assert isinstance(cnn_json_dict, dict), 'cnn_json_dict should be dict, while the input is {}'.format(type(cnn_json_dict))
    cnn_json_struct = JsonFormatSetting.CNN_JSON_STRUCTURE

    return check_dict(cnn_json_dict, cnn_json_struct)


def labelme_json_check(labelme_json_dict):
    """
    labelme_json dict check
    --- a dictionary is required as the input ---
    """
    assert isinstance(labelme_json_dict, dict), 'labelme_json_dict should be dict, while the input is {}'.format(type(labelme_json_dict))
    labelme_json_struct = JsonFormatSetting.LABELME_JSON_STRUCTURE

    return check_dict(labelme_json_dict, labelme_json_struct)
