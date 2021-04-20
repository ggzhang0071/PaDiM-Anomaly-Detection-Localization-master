timestamp=`date +%Y%m%d%H%M%S`
rm Logs/*.log
#rm kangqiang_result/pictures_wide_resnet50_2/*
#rm kangqiang_result/segment_image_result_wide_resnet50_2/*
#product_class='0808QFN-16L' 'DFN-002A' 'QFN-3X3-16L' '0708DFN-8L' 'DFN-5X6-T-8L' '0420QFN-5X6-8L' '1101QFN-40L' '0713DFN-2X3-8L' 'DFN-5X6-8L' '1129QFN-4X4-24L'



for product_id in "0808QFN-16L" "DFN-002A" "QFN-3X3-16L"
do
    python3  main_padim.py --train_num_samples  1500  --test_num_samples 1500 --product_class $product_id  2>&1 |tee Logs/$timestamp.log
done
