"""Module containing the path to the data, the logs and the model weights
"""
import os

FASTMRI_DATA_DIR = os.environ.get('FASTMRI_DATA_DIR', '/media/Zaccharie/UHRes/')
OASIS_DATA_DIR = os.environ.get('OASIS_DATA_DIR', '/media/Zaccharie/UHRes/')
LOGS_DIR = os.environ.get('LOGS_DIR', './')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', './')
USE_BRAIN_DATA = int(os.environ.get('USE_BRAIN_DATA', 0))


directory_names = {
    'brain': {
        'train': 'brain_multicoil_train',
        'val': 'brain_multicoil_val',
        'test': 'brain_multicoil_test',
    },
    'multicoil': {
        'train': 'multicoil_train',
        'val': 'multicoil_val',
        'test': 'multicoil_test',
    },
    'singlecoil': {
        'train': 'singlecoil_train/singlecoil_train',
        'val': 'singlecoil_val',
        'test': 'singlecoil_test_v2',
    },
}
