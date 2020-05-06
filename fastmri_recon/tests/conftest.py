import h5py
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from tqdm import tqdm


K_shape_single_coil = (2, 640, 322)
K_shape_multi_coil = (2, 15, 640, 322)
I_shape = (2, 320, 320)
contrast = 'CORPD_FBK'

def create_data(filename, multicoil=False):
    k_shape = K_shape_single_coil
    image_ds = "reconstruction_esc"
    if multicoil:
        k_shape = K_shape_multi_coil
        image_ds = "reconstruction_rss"
    kspace = np.random.normal(size=k_shape) + 1j * np.random.normal(size=k_shape)
    image = np.random.normal(size=I_shape)
    kspace = kspace.astype(np.complex64)
    image = image.astype(np.float32)
    with h5py.File(filename, "w") as h5_obj:
        h5_obj.create_dataset("kspace", data=kspace)
        h5_obj.create_dataset(image_ds, data=image)
        h5_obj.attrs['acquisition'] = contrast

@pytest.fixture(scope="session")
def monkeysession(request):
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()

@pytest.fixture(scope="session", autouse=True)
def create_full_fastmri_test_tmp_dataset(monkeysession, tmpdir_factory):
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
    n_files = 2
    # train
    for i in tqdm(range(n_files), 'Creating single coil train files'):
        data_filename = f"train_singlecoil_{i}.h5"
        create_data(str(fastmri_tmp_singlecoil_train.join(data_filename)))
    # val
    for i in tqdm(range(n_files), 'Creating single coil val files'):
        data_filename = f"val_singlecoil_{i}.h5"
        create_data(str(fastmri_tmp_singlecoil_val.join(data_filename)))
    #### multi coil
    fastmri_tmp_multicoil_train = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('multicoil_train')
    ), numbered=False)
    fastmri_tmp_multicoil_val = tmpdir_factory.mktemp(str(
        fastmri_tmp_data_dir.join('multicoil_val')
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
    return {
        'fastmri_tmp_data_dir': str(fastmri_tmp_data_dir) + '/',
        'logs_tmp_dir': str(tmpdir_factory.getbasetemp()) + '/',
        'checkpoints_tmp_dir': str(tmpdir_factory.getbasetemp()) + '/',
        'fastmri_tmp_singlecoil_train': str(fastmri_tmp_singlecoil_train) + '/',
        'fastmri_tmp_singlecoil_val': str(fastmri_tmp_singlecoil_val) + '/',
        'fastmri_tmp_multicoil_train': str(fastmri_tmp_multicoil_train) + '/',
        'fastmri_tmp_multicoil_val': str(fastmri_tmp_multicoil_val) + '/',
        'K_shape_single_coil': K_shape_single_coil,
        'K_shape_multi_coil': K_shape_multi_coil,
        'I_shape': I_shape,
        'contrast': contrast,
    }
