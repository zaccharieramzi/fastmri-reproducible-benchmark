"""Module containing the path to the data, the logs and the model weights
"""
import os

FASTMRI_DATA_DIR = os.environ.get('FASTMRI_DATA_DIR', '/media/Zaccharie/UHRes/')
OASIS_DATA_DIR = os.environ.get('OASIS_DATA_DIR', '/media/Zaccharie/UHRes/')
LOGS_DIR = os.environ.get('LOGS_DIR', './')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', './')

brain_volumes_per_contrast = {
    'validation':{
        'AXFLAIR': 107,
        'AXT1POST': 287,
        'AXT1PRE': 77,
        'AXT1': 92,
        'AXT2': 815,
    },
    'test': {
        'AXFLAIR': 49,
        'AXT1POST': 122,
        'AXT1PRE': 36,
        'AXT1': 29,
        'AXT2': 322,
    }
}
