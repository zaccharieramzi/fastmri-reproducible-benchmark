import h5py
import numpy as np
import pytest
from tqdm import tqdm


K_shape_single_coil = (2, 640, 322)
K_shape_multi_coil = (2, 15, 640, 322)
I_shape = (2, 320, 320)
contrast = 'CORPD_FBK'

def create_data(filename, multicoil=False, train=True):
    k_shape = K_shape_single_coil
    if train:
        image_ds = "reconstruction_esc"
        if multicoil:
            k_shape = K_shape_multi_coil
            image_ds = "reconstruction_rss"
        image = np.random.normal(size=I_shape)
        image = image.astype(np.float32)
    else:
        mask_shape = [K_shape_multi_coil[-1]]
        mask = np.random.choice(a=[True, False], size=mask_shape)
        af = np.sum(mask.astype(int)) / mask_shape[0]
    kspace = np.random.normal(size=k_shape) + 1j * np.random.normal(size=k_shape)
    kspace = kspace.astype(np.complex64)
    with h5py.File(filename, "w") as h5_obj:
        h5_obj.create_dataset("kspace", data=kspace)
        if train:
            h5_obj.create_dataset(image_ds, data=image)
        else:
            h5_obj.create_dataset('mask', data=mask)
        h5_obj.attrs['acquisition'] = contrast
    return af


@pytest.fixture(scope="session", autouse=False)
def create_full_fastmri_test_tmp_dataset(tmpdir_factory):
    # main dirs
    fastmri_tmp_data_dir = tmpdir_factory.mktemp(
        "fastmri_test_tmp_data",
        numbered=False,
    )
    logs_tmp_dir = tmpdir_factory.mktemp(
        "logs",
        numbered=False,
    )
    checkpoints_tmp_dir = tmpdir_factory.mktemp(
        "checkpoints",
        numbered=False,
    )
    #### single coil
    fastmri_tmp_singlecoil_train = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('singlecoil_train')
    ), numbered=False)
    fastmri_tmp_singlecoil_train = tmpdir_factory.mktemp(str(
        fastmri_tmp_singlecoil_train.join('singlecoil_train')
    ), numbered=False)
    fastmri_tmp_singlecoil_val = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('singlecoil_val')
    ), numbered=False)
    fastmri_tmp_singlecoil_test = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('singlecoil_test')
    ), numbered=False)
    n_files = 2
    # train
    for i in tqdm(range(n_files), 'Creating single coil train files'):
        data_filename = f"train_singlecoil_{i}.h5"
        create_data(str(fastmri_tmp_singlecoil_train.join(data_filename)))
    # val
    for i in tqdm(range(n_files), 'Creating single coil val files'):
        data_filename = f"val_singlecoil_{i}.h5"
        create_data(str(fastmri_tmp_singlecoil_val.join(data_filename)))
    # test
    af_single_coil = []
    for i in tqdm(range(n_files), 'Creating single coil test files'):
        data_filename = f"test_singlecoil_{i}.h5"
        af = create_data(
            str(fastmri_tmp_singlecoil_test.join(data_filename)),
            multicoil=False,
            train=False,
        )
        af_single_coil.append(af)
    #### multi coil
    fastmri_tmp_multicoil_train = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('multicoil_train')
    ), numbered=False)
    fastmri_tmp_multicoil_val = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('multicoil_val')
    ), numbered=False)
    fastmri_tmp_multicoil_test = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('multicoil_test')
    ), numbered=False)
    n_files = 2
    # train
    for i in tqdm(range(n_files), 'Creating multi coil train files'):
        data_filename = f"train_multicoil_{i}.h5"
        create_data(
            str(fastmri_tmp_multicoil_train.join(data_filename)),
            multicoil=True,
        )
    # val
    for i in tqdm(range(n_files), 'Creating multi coil val files'):
        data_filename = f"val_multicoil_{i}.h5"
        create_data(
            str(fastmri_tmp_multicoil_val.join(data_filename)),
            multicoil=True,
        )
    # test
    af_multi_coil = []
    for i in tqdm(range(n_files), 'Creating multi coil test files'):
        data_filename = f"test_multicoil_{i}.h5"
        af = create_data(
            str(fastmri_tmp_multicoil_test.join(data_filename)),
            multicoil=True,
            train=False,
        )
        af_multi_coil.append(af)

    return {
        'fastmri_tmp_data_dir': str(fastmri_tmp_data_dir) + '/',
        'logs_tmp_dir': str(tmpdir_factory.getbasetemp()) + '/',
        'checkpoints_tmp_dir': str(tmpdir_factory.getbasetemp()) + '/',
        'fastmri_tmp_singlecoil_train': str(fastmri_tmp_singlecoil_train) + '/',
        'fastmri_tmp_singlecoil_val': str(fastmri_tmp_singlecoil_val) + '/',
        'fastmri_tmp_singlecoil_test': str(fastmri_tmp_singlecoil_test) + '/',
        'fastmri_tmp_multicoil_train': str(fastmri_tmp_multicoil_train) + '/',
        'fastmri_tmp_multicoil_val': str(fastmri_tmp_multicoil_val) + '/',
        'fastmri_tmp_multicoil_test': str(fastmri_tmp_multicoil_test) + '/',
        'af_single_coil': af_single_coil,
        'af_multi_coil': af_multi_coil,
        'K_shape_single_coil': K_shape_single_coil,
        'K_shape_multi_coil': K_shape_multi_coil,
        'I_shape': I_shape,
        'contrast': contrast,
    }
