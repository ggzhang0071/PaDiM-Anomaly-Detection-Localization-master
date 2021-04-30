timestamp=`date +%Y%m%d%H%M%S`
rm Logs/*.log 

python generate_json_for_classification.py  
 2>&1 |tee Logs/$timestamp.log

 python generate_image_label_for_classification.py  
 2>&1 |tee Logs/$timestamp.log


