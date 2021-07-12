
import os
import glob
import shutil

import numpy as np

from container.modeling.config import DATA_LOCAL_DIR, PERCENTAGE_VAL,\
    PERCENTAGE_EVAL, RANDOM_SEED


def split_data_into_class_folders():
    """
    Shuffle the images in each classe folder and split the dataset in 3 folders:
    - training
    - validation
    - evaluation
    """
    # create the three sub-dataset folder paths
    path_to_train_data = os.path.join(DATA_LOCAL_DIR, 'training')
    path_to_val_data = os.path.join(DATA_LOCAL_DIR, 'validation')
    path_to_eval_data = os.path.join(DATA_LOCAL_DIR, 'evaluation')
    # Loop over the classes folders
    for class_name in os.listdir(DATA_LOCAL_DIR):
        print(class_name)
        if class_name not in ['training', 'validation', 'evaluation']:
            # List all the images in the current class folder 
            class_path = os.path.join(DATA_LOCAL_DIR, class_name)
            print(class_path)
            imgs_paths = glob.glob(class_path + '/*.jpg')
            nb_images = len(imgs_paths)
            # Compute the index limites for each type of data
            train_lim = int(nb_images * (1 - PERCENTAGE_VAL - PERCENTAGE_EVAL))
            val_lim = train_lim + int(nb_images * PERCENTAGE_VAL)
            # print the index limites
            print(train_lim, val_lim, nb_images)
            # Shuffle the images
            np.random.shuffle(imgs_paths)
            # Store the training images in the training folder
            for path in imgs_paths[:train_lim]:
                basename = os.path.basename(path)
                path_to_save = os.path.join(path_to_train_data, class_name)
                # Build the intermediate folder when necessary
                if not os.path.isdir(path_to_save):
                    os.makedirs(path_to_save)
                 # Copy the image in the coresponfing folder
                shutil.copy(path, os.path.join(path_to_save, basename))
            # Store the validation images in the validation folder
            for path in imgs_paths[train_lim:val_lim]:
                basename = os.path.basename(path)
                path_to_save = os.path.join(path_to_val_data, class_name)
                # Build the intermediate folder when necessary
                if not os.path.isdir(path_to_save):
                    os.makedirs(path_to_save)
                # Copy the image in the coresponfing folder
                shutil.copy(path, os.path.join(path_to_save, basename))
            # Store the evaluation images in the evaluation folder
            for path in imgs_paths[val_lim:]:
                basename = os.path.basename(path)
                path_to_save = os.path.join(path_to_eval_data, class_name)
                # Build the intermediate folder when necessary
                if not os.path.isdir(path_to_save):
                    os.makedirs(path_to_save)
                # Copy the image in the coresponfing folder
                shutil.copy(path, os.path.join(path_to_save, basename))

if __name__ == '__main__':

    split_data_switch = True
    # Set the random seed
    np.random.seed(RANDOM_SEED)

    if split_data_switch :
        split_data_into_class_folders()
