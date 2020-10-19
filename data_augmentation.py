import os
import cv2
import numpy as np


def data_aug_translation(img_folder_path):
    img_paths = os.listdir(img_folder_path)
    num = len(img_paths) + 1
    for img_path in img_paths:
        img = cv2.imread(img_folder_path+'/'+img_path)
        img_size = img.shape[0]
        r1, r2 = np.random.rand(1,2)[0] - 0.5
        move1 = np.array([[1, 0, img_size*r1], [0, 1, img_size*r2]], dtype=np.float32)

def data_aug_rotation(img_folder_path):
    img_paths = os.listdir(img_folder_path)
    num = len(img_paths) + 1
    for img_path in img_paths:
        img = cv2.imread(img_folder_path+'/'+img_path)
        rows, cols = img.shape[:2]
        for angle in [90, 180, 270]:
            move = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
            img_change = cv2.warpAffine(img, move, (cols,rows))
            cv2.imwrite(img_folder_path+'/z_augment_%d.jpg'%(num), img_change)
            num += 1


# '/home/qiangde/PycharmProjects/utils/train/01'
if __name__ == '__main__':
    # data_aug_rotation('/home/qiangde/PycharmProjects/utils/train/01')
    r1, r2 = np.random.rand(1,2)[0]-0.5
    print(r1, ' ', r2)
























