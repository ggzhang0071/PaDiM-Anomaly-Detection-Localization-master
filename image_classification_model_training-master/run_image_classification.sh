
rm Logs/image_classification_model_training_*.log 

for i  in {1..5}
do  
timestamp=`date +%Y%m%d%H%M%S`
python train.py  2>&1 |tee ../Logs/image_classification_model_training_$timestamp.log
done

