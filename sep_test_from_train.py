import os
import random


# satisfy out/train/[class1, class2, ...]
def seperate(ratio=0.2):
    class_names = os.listdir('out/train')
    for class_name in class_names:
        train_path = os.path.join('out/train', class_name)
        test_path = os.path.join('out/test', class_name)
        os.makedirs(test_path)
        train_samples = os.listdir(train_path)
        test_samples = random.sample(train_samples, int(len(train_samples)*ratio))
        for test_sample in test_samples:
            os.rename(os.path.join(train_path, test_sample), os.path.join(test_path, test_sample))

def cancel():
    for class_name in os.listdir('out/test'):
        current_folder = os.path.join('out/test', class_name)
        target_folder = os.path.join('out/train', class_name)
        imgs = os.listdir(current_folder)
        for img in imgs:
            os.rename(os.path.join(current_folder, img), os.path.join(target_folder, img))

if __name__ == '__main__':
    seperate()


