"""Module containing the path to the data, the logs and the model weights
"""
import os

FASTMRI_DATA_DIR = os.environ.get('FASTMRI_DATA_DIR', '/media/Zaccharie/UHRes/')
OASIS_DATA_DIR = os.environ.get('OASIS_DATA_DIR', '/media/Zaccharie/UHRes/OASIS_data')
LOGS_DIR = os.environ.get('LOGS_DIR', './')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', './')
TMP_DIR = os.environ.get('TMP_DIR', './')

n_volumes_train = 973
n_volumes_val = 199
n_volumes_test = {
    4: 50,
    8: 58,
}

brain_volumes_per_contrast = {
    'train': {
        'AXFLAIR': 344,
        'AXT1POST': 949,
        'AXT1PRE': 250,
        'AXT1': 248,
        'AXT2': 2678,
    },
    'validation': {
        'AXFLAIR': 107,
        'AXT1POST': 287,
        'AXT1PRE': 77,
        'AXT1': 92,
        'AXT2': 815,
    },
    'test': {
        4: {
            'AXFLAIR': 24,
            'AXT1POST': 54,
            'AXT1PRE': 17,
            'AXT1': 16,
            'AXT2': 170
        },
        8: {
            'AXFLAIR': 25,
            'AXT1POST': 68,
            'AXT1PRE': 19,
            'AXT1': 13,
            'AXT2': 152
        },
    }
}

brain_n_volumes_train = 4469
brain_n_volumes_validation = 1378
brain_n_volumes_test = {
    4: 281,
    8: 277,
}
