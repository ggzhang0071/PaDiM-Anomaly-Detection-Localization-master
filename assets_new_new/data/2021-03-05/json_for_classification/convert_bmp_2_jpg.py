import os,cv2
def convert_bmp_to_jpg(file_path):
    image_name_list=os.listdir(file_path)
    for image_name in image_name_list:
        if image_name.endswith(".bmp"):
            img=cv2.imread(os.path.join(file_path,image_name))
            if not img is None:
                image_name_without_ext=os.path.splitext(image_name)[0]
                cv2.imwrite(os.path.join(file_path,image_name_without_ext+".jpg"),img)

if __name__ == '__main__':
    file_path="/git/PaDiM-master/kangqiang_result/croped_images_part_with_classification/7"
    convert_bmp_to_jpg(file_path)


        




