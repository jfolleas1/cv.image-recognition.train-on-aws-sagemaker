import os
import config as cfg

def get_number_of_imgs_inside_folder(directory):

    totalcount = 0

    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in [".png", ".jpg", ".jpeg"]:
                totalcount = totalcount + 1

    return totalcount

def get_nb_step(data_dir, data_type):

    assert data_type in ['training', 'validation', 'evaluation']

    if data_type == 'training':
        config_nb_steps = cfg.STEP_PER_EPOCH
    elif data_type == 'validation':
        config_nb_steps = cfg.VALIDATION_STEPS
    elif data_type == 'evaluation':
        config_nb_steps = cfg.EVALUATION_STEPS
    
    # If the config is set at auto, we use the total number of images 
    # divided by the batch size
    if config_nb_steps == 'auto':
        path_data_type_folder = os.path.join(data_dir, data_type)

        total_data_type_imgs = get_number_of_imgs_inside_folder(\
            path_data_type_folder)

        config_nb_steps = total_data_type_imgs // cfg.BATCH_SIZE
    
    return config_nb_steps
