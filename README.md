# fastMRI reproducible benchmark

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zaccharieramzi/fastmri-reproducible-benchmark/binder)

The idea of this repository is to have a way to rapidly benchmark new solutions against existing reconstruction algorithms on the fastMRI dataset single-coil track.
The reconstruction algorithms implemented or adapted to the fastMRI dataset include to this day:
- zero filled reconstruction
- [LORAKS](https://www.ncbi.nlm.nih.gov/pubmed/24595341), using the [LORAKS Matlab toolbox](https://mr.usc.edu/download/LORAKS2/)
- Wavelet-based reconstruction (i.e. solving an L1-based analysis formulation optimisation problem with greedy FISTA), using [pysap-mri](https://github.com/CEA-COSMIC/pysap-mri)
- U-net
- [DeepCascade net](https://arxiv.org/abs/1704.02422)
- [KIKI net](https://www.ncbi.nlm.nih.gov/pubmed/29624729)
- [Learned Primal Dual](https://arxiv.org/abs/1707.06474), adapted to MRI reconstruction

All the neural networks (except the U-net) are implemented in both `keras` and `pytorch`.
I mainly used `keras` to develop, but I realized at some point that `pytorch` might just be faster for fourier transform operations (see https://github.com/tensorflow/tensorflow/issues/6541).
However, the main documentation is still for the `keras` models.

## Current results
The current results are not in-line with the results the authors obtained in their papers (and obtained by others on other databases), that is, the U-net is for now the best performer.
For LORAKS, the author highlighted the fact that the mask employed in the fastMRI dataset is not adapted to the algorithm. Moreover, the kspace zero-padding, and the oversampled AD strongly impede the performance of LORAKS.

For the neural networks, it's still unclear why they don't perform well on this data. Regarding the mask, I tried with a more common variable density scheme, but it didn't improve the results. Some authors have suggested that knee data might be more difficult to reconstruct.

## How to train the neural networks
The scripts to train the neural networks are located in `fastmri_recon/training/`.
You just need to install the package and its dependencies:
```
pip install . &&\
pip install -r fastmri_recon/requirements.txt
```
TensorFlow is not listed as a dependency to let you chose if you want gpu supported TensorFlow.


## How to write a new neural network for reconstruction
A good example of a simple neural network on which you can improve is the `zerofill_net` which is simply performing zero-filled reconstruction using `keras`.
The building blocks can then be found in `fastmri_recon/helpers/nn_mri.py`
