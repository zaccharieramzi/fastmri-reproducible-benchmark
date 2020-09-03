from pathlib import Path

from ismrmrd import Dataset
import numpy as np
from tqdm import tqdm

from fastmri_recon.config import FASTMRI_DATA_DIR
from fastmri_recon.data.utils.crop import crop_center
from fastmri_recon.data.utils.h5 import from_multicoil_train_file_to_image
from fastmri_recon.data.utils.ismrmrd import from_fastmri_to_ismrmrd
from fastmri_recon.evaluate.metrics.np_metrics import METRIC_FUNCS, Metrics
from fastmri_recon.evaluate.reconstruction.gadgetron_reconstruction import gadgetron_grappa_reconstruction, GRAPPA_RECON_DS_NAME


def corresponding_volume(filename):
    return filename.name.split('_')[0]

def get_slice(filename):
    slice_ds = Dataset(
        filename,
        dataset_name=GRAPPA_RECON_DS_NAME,
        create_if_needed=False,
    )
    im = slice_ds.read_image('image_1', 0).data
    im = crop_center(np.transpose(np.squeeze(im.data)), 320)
    return im

def generate_ismrmrd_files(af=4, split='val', organ='knee'):
    original_directory = f'multicoil_{split}'
    if organ == 'brain':
        original_directory = 'brain_' + original_directory
    original_directory = Path(FASTMRI_DATA_DIR) / original_directory
    ismrmrd_dir = Path(FASTMRI_DATA_DIR) / f'{split}_{organ}_{af}_ismrmrd/'
    ismrmrd_dir.mkdir(exist_ok=True)
    filenames = sorted(list(original_directory.glob('*.h5')))
    for f in tqdm(filenames):
        from_fastmri_to_ismrmrd(f, out_dir=ismrmrd_dir)

def eval_gadgetron(af=4, split='val', organ='knee', my_config=False, n_volumes=50):
    original_directory = f'multicoil_{split}'
    if organ == 'brain':
        original_directory = 'brain_' + original_directory
    original_directory = Path(FASTMRI_DATA_DIR) / original_directory
    ismrmrd_dir = Path(FASTMRI_DATA_DIR) / f'{split}_{organ}_{af}_ismrmrd/'
    ismrmrd_out_dir = ismrmrd_dir.parent / (ismrmrd_dir.name + '_results')
    filenames = sorted(list(ismrmrd_dir.glob('*.h5')))
    current_volume = None
    current_volume_slices = []
    i_volume = 0
    m = Metrics(METRIC_FUNCS)
    for f in tqdm(filenames):
        volume = corresponding_volume(f)
        out_f = ismrmrd_out_dir / 'out' + f.name
        gadgetron_grappa_reconstruction(f, out_f, my_config)
        if current_volume != volume:
            # make a volume out of the recons
            # get the corresponding gt from fastmri
            # reset current_volume to the new volume
            if current_volume_slices:
                recon_volume = np.array(current_volume_slices)
                corresponding_fastmri_file = original_directory / (volume + '.h5')
                gt_volume = from_multicoil_train_file_to_image(corresponding_fastmri_file)
                m.push(gt_volume, recon_volume)
                i_volume += 1
                if i_volume >= n_volumes:
                    break
            current_volume_slices = []
            current_volume = volume
        # append the slice to the volume
        recon_slice = get_slice(out_f)
        current_volume_slices.append(recon_slice)

    # do one final evaluation for the last volume
    if i_volume < n_volumes:
        recon_volume = np.array(current_volume_slices)
        corresponding_fastmri_file = original_directory / (volume + '.h5')
        gt_volume = from_multicoil_train_file_to_image(corresponding_fastmri_file)
        m.push(gt_volume, recon_volume)

    print(m)
    return m
