## Repo for image classification training

This is a repo of a training pipeline for image classification models. There are two training scripts in the repo <em>train.py</em> and <em>train_w_unsuprivised_ood.py</em>. Use train.py for vanilla training, use train_w_unsupervised_ood.py if you have access to OOD(Out-of-Distribution) data and would like to train models that are robust to OOD data.

to customize training, specify the training details in <em>config.json</em>.

```json
{
    "epochs": 300,
    "batch_size": 128,
    "class_num": 8,
    "input_size": 128,
    "embedding_dim": 64,
    "classes":  {"0": "shakong", "1": "guoshi", "2": "yiwu", "3": "feilinjiao", "4": "huashang_yin", "5": "huashang_tong", "6": "yin_wuzi", "7": "tong_wuzi"},
    "device_id": 1,
    "model": "resnet50",
    "test_path": "/path/to/test/txt",
    "train_path": "/path/to/train/txt",
    "val_path": "/path/to/val/",
    "save_path": "/save/path",
    "ood_train_path": "/path/to/out/of/distribution/train/data/if_any",
    "ood_val_path": "/path/to/out/of/distribution/val/data/if_any"
}
```
and train with <em>train.py</em> or <em>train_W_unsupervised</em>. Leave "ood_train_path" and "ood_val_path" blank if using vanilla training.

to train model with OOD data, sepcify <em>ood_train_path</em> and <em>ood_val_path</em> in <em>config.json</em>.

