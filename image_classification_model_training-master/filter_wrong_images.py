import os,glob
import cv2
image_data_root="/git/PaDiM-master/kangqiang_result/croped_images/**/*.jpg"
for image_path in glob.glob(image_data_root,recursive=True):
    img = cv2.imread(image_path)
    if img is None:
        print(image_path)
        os.remove(image_path)

    


