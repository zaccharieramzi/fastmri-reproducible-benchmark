"""Module containing the path to the data, the logs and the model weights
"""
import os

FASTMRI_DATA_DIR = os.environ.get('FASTMRI_DATA_DIR', '/neurospin/optimed/zramzi/fastMRI_data/')
OASIS_DATA_DIR = os.environ.get('OASIS_DATA_DIR', '/media/Zaccharie/UHRes/')
LOGS_DIR = os.environ.get('LOGS_DIR', './')
CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', './')
