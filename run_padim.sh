timestamp=`date +%Y%m%d%H%M%S`
rm Logs/*.log
rm kangqiang_result/pictures_wide_resnet50_2/*

python main.py --train_num_samples 1000 --test_num_samples 1000  2>&1 |tee Logs/$timestamp.log