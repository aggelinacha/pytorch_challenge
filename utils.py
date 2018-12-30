"""!
@brief Utilities

@author Aggelina Chatziagapi {aggelina.cha@gmail.com}
"""

import json
import torch


def get_label_mapping(input_json="cat_to_name.json"):
    """!
    @brief Read json file with label mapping (from category label to
        category name).
    """
    with open(input_json, "r") as f:
        cat_to_name = json.load(f)

    return cat_to_name


def check_cuda():
    """!
    @brief Check if CUDA is available.
    """
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    return train_on_gpu
