from tqdm import tqdm

from fastmri_recon.data.utils.h5 import from_multicoil_train_file_to_image_and_kspace_and_contrast

test_file_multi_coil = 'fastmri_recon/tests/fastmri_data/multi_coil/file1000001.h5'

for i in tqdm(range(10000)):
    res = from_multicoil_train_file_to_image_and_kspace_and_contrast(
        test_file_multi_coil,
        [{'inner_slices': 8, 'rand': True}],
    )
