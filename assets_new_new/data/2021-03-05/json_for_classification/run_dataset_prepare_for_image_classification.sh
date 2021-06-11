
rm /git/PaDiM-master/assets_new_new/data/2021-03-05/json_for_classification/data_prepare_for_semi_supervised_learning/*.txt


python3 get_train_val_test_dataset_from_the_classification_files.py --choose_labels 0 3  9
