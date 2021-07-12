import os
import config as cfg

def get_number_of_imgs_inside_folder(directory):
    """
    Count the number of images in the indicated folder.

    Params:
        - directory (str): The path to the directory of which we want to count
            the number of pictures within it.

    Returns:
        - totalcount (int): The number of picutres within the directory.
    """
    totalcount = 0
    # Loop over the filenames in the directory
    for _, _, filenames in os.walk(directory):
        for filename in filenames:
            # Only conside the files with an image extention
            _, ext = os.path.splitext(filename)
            if ext in [".png", ".jpg", ".jpeg"]:
                totalcount = totalcount + 1
    return totalcount

def get_nb_step(data_dir, data_type):
    """
    Compute the number of step per epoch or per validaiton/evaluation phase 
    base on the total number of images availables in each dataset if not 
    indicated in the configuration. If indicated in the configuration, use the 
    indicated number.

    Params:
        - data_dir (str): The path to the directory containing the images
        - data_type (str): The type of data we want to compute the number of 
            step of. Must be one of the three ['training', 'validation',
            'evaluation'].

    Returns:
        config_nb_steps (int): The number of steps requested.
    """

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
