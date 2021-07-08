import glob
import shutil
import os
import random
from PIL import Image
import numpy as np

from container.modeling.config import DATA_LOCAL_DIR, PERCENTAGE_VAL, PERCENTAGE_EVAL


def split_data_into_class_folders():

    path_to_train_data = os.path.join(DATA_LOCAL_DIR, 'training')
    path_to_val_data = os.path.join(DATA_LOCAL_DIR, 'validation')
    path_to_eval_data = os.path.join(DATA_LOCAL_DIR, 'evaluation')

    for class_name in os.listdir(DATA_LOCAL_DIR):
        print(class_name)
        if class_name not in ['training', 'validation', 'evaluation']:
            class_path = os.path.join(DATA_LOCAL_DIR, class_name)
            print(class_path)
            imgs_paths = glob.glob(class_path + '/*.jpg')
            nb_images = len(imgs_paths)

            train_lim = int(nb_images * (1 - PERCENTAGE_VAL - PERCENTAGE_EVAL))
            val_lim = train_lim + int(nb_images * PERCENTAGE_VAL)

            print(train_lim, val_lim, nb_images)

            for path in imgs_paths[:train_lim]:
                basename = os.path.basename(path)
                path_to_save = os.path.join(path_to_train_data, class_name)
                if not os.path.isdir(path_to_save):
                    os.makedirs(path_to_save)
                shutil.copy(path, os.path.join(path_to_save, basename))
            
            for path in imgs_paths[train_lim:val_lim]:
                basename = os.path.basename(path)
                path_to_save = os.path.join(path_to_val_data, class_name)
                if not os.path.isdir(path_to_save):
                    os.makedirs(path_to_save)
                shutil.copy(path, os.path.join(path_to_save, basename))

            for path in imgs_paths[val_lim:]:
                basename = os.path.basename(path)
                path_to_save = os.path.join(path_to_eval_data, class_name)
                if not os.path.isdir(path_to_save):
                    os.makedirs(path_to_save)
                shutil.copy(path, os.path.join(path_to_save, basename))





if __name__ == '__main__':

    split_data_switch = True


    if split_data_switch :
        split_data_into_class_folders()
