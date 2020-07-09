from builtins import FileExistsError
from pathlib import Path
import random
import re
import shutil

from tqdm import tqdm

from fastmri_recon.config import OASIS_DATA_DIR


def _get_subject_from_filename(filename):
    base_name = filename.name
    subject_id = re.findall(r'OAS3\d{4}', base_name)[0]
    return subject_id

def split_oasis(val_split=0.1, seed=0):
    filenames = list(Path(OASIS_DATA_DIR).rglob('*.nii.gz'))
    subjects = [_get_subject_from_filename(filename) for filename in filenames]
    unique_subjects = list(set(subjects))
    n_val = int(len(unique_subjects) * val_split)
    random.seed(seed)
    random.shuffle(unique_subjects)
    val_subjects = subjects[:n_val]
    val_filenames = [filename for filename in filenames if _get_subject_from_filename(filename) in val_subjects]
    train_filenames = [filename for filename in filenames if filename not in val_filenames]
    return train_filenames, val_filenames

def make_split_dir(val_split=0.1, seed=0):
    try:
        train_dir = Path(OASIS_DATA_DIR) / 'train'
        train_dir.mkdir(parents=False, exist_ok=False)
        val_dir = Path(OASIS_DATA_DIR) / 'val'
        val_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        print(
            """Train validation split was already carried, if you want to do it
            again, please delete the split directories."""
        )
        raise
    train_filenames, val_filenames = split_oasis(val_split=val_split, seed=seed)
    for train_filename in tqdm(train_filenames, desc='train files'):
        copied_filename = train_dir / train_filename.name
        shutil.copy(train_filename, copied_filename)
    for val_filename in tqdm(val_filenames, desc='val files'):
        copied_filename = val_dir / val_filename.name
        shutil.copy(val_filename, copied_filename)



if __name__ == '__main__':
    make_split_dir()
