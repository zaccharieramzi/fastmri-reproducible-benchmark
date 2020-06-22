from tqdm import tqdm

from fastmri_recon.config import *
from fastmri_recon.data.datasets.multicoil.fastmri_pyfunc import test_masked_kspace_dataset_from_indexable, test_filenames
from fastmri_recon.evaluate.reconstruction.grappa_reconstruction import reco_grappa
from ..utils.write_results import write_result


def grappa_inference(contrast=None, af=4, n_samples=None, exp_id='grappa'):
    test_path = f'{FASTMRI_DATA_DIR}multicoil_test_v2/'

    af = int(af)

    test_set = test_masked_kspace_dataset_from_indexable(
        test_path,
        AF=af,
        contrast=contrast,
        scale_factor=1e6,
        n_samples=n_samples,
    ).as_numpy_iterator()
    test_set_filenames = test_filenames(
        test_path,
        AF=af,
        contrast=contrast,
        n_samples=n_samples,
    ).as_numpy_iterator()
    tqdm_total = 30 if n_samples is None else n_samples
    for (kspace, _, _), filename in tqdm(zip(test_set, test_set_filenames), total=tqdm_total):
        res = reco_grappa(kspace[..., 0])
        write_result(exp_id, res[..., None], filename.decode('utf-8'))
