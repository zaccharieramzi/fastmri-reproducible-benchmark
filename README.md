# fastMRI reproducible benchmark

[![Build status](https://travis-ci.com/zaccharieramzi/fastmri-reproducible-benchmark.svg?branch=master)](https://travis-ci.org/zaccharieramzi/fastmri-reproducible-benchmark)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zaccharieramzi/fastmri-reproducible-benchmark/master)

The idea of this repository is to have a way to rapidly benchmark new solutions against existing reconstruction algorithms on the fastMRI dataset single-coil track.
The reconstruction algorithms implemented or adapted to the fastMRI dataset include to this day:
- zero filled reconstruction
- [LORAKS](https://www.ncbi.nlm.nih.gov/pubmed/24595341), using the [LORAKS Matlab toolbox](https://mr.usc.edu/download/LORAKS2/)
- Wavelet-based reconstruction (i.e. solving an L1-based analysis formulation optimisation problem with greedy FISTA), using [pysap-mri](https://github.com/CEA-COSMIC/pysap-mri)
- U-net
- [DeepCascade net](https://arxiv.org/abs/1704.02422)
- [KIKI net](https://www.ncbi.nlm.nih.gov/pubmed/29624729)
- [Learned Primal Dual](https://arxiv.org/abs/1707.06474), adapted to MRI reconstruction
- [XPDNet](https://arxiv.org/abs/2010.07290), a modular unrolled reconstruction algorithm, in which you can plug your best denoiser.

All the neural networks are implemented in TensorFlow with the Keras API.
The older ones (don't judge this was the beginning of my thesis) are coded using the functional API.
The more recent ones are coded in the subclassed API and are more modular.
For the LORAKS reconstruction, you will not be able to reconstruct the proper fastMRI data because of the A/D oversampling.

## Reconstruction settings

The main reconstruction settings are cartesian single-coil and multi-coil 2D reconstruction with random and "periodic sampling".
These settings are covered by almost all the networks in this repo, mainly because they are the settings of the fastMRI challenge.

### Other reconstruction settings

*Non-cartesian*: you can reconstruct non-cartesian data using the [NCPDNet](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/blob/master/fastmri_recon/models/subclassed_models/ncpdnet.py).
It relies on the TensorFlow implementation of the NUFFT, [`tfkbnufft`](https://github.com/zaccharieramzi/tfkbnufft).


## How to train the neural networks
The scripts to train the neural networks are located in `fastmri_recon/training_scripts/`.
You just need to install the package and its dependencies:
```
pip install . &&\
pip install -r requirements.txt
```


## How to write a new neural network for reconstruction
The simplest and most versatile way to write a neural network for reconstruction is to subclass the [`CrossDomainNet` class](fastmri_recon/models/subclassed_models/cross_domain.py).
An example is the [`PDnet`](fastmri_recon/models/subclassed_models/pdnet.py)

## Note on reproducibility
Because of changes to the U-net design, the checkpoints for the U-net are not valid for the latest version of the source code (see this [issue](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/issues/104)).
To reproduce the results you can checkout to the `bcd3fdd` commit (`git checkout bcd3fdd`).
I am working on a way to make sure this is not needed in the future, by providing an old implementation.

To reproduce the results of the Benchmarking papers, you will need to use the [`validation_for_net`](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/blob/master/experiments/validation_for_net.ipynb) notebook.
I am also working on a script to make this more seamless.

Finally to reproduce the figures of the Benchmarking papers, you will need to use the [`qualitative_validation_for_net`](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/blob/master/experiments/qualitative_validation_for_net.ipynb) notebook.

# Data requirements

## fastMRI

The fastMRI data must be located in a directory whose path is stored in the `FASTMRI_DATA_DIR` environment variable.
It can be downloaded on [the official website](https://fastmri.med.nyu.edu/) after submitting a request (bottom of the page).

The package currently supports public single coil and multi coil knee data.

## OASIS

The OASIS data must be located in a directory whose path is stored in the `OASIS_DATA_DIR` environment variable.
It can be downloaded on [the XNAT store](https://central.xnat.org/app/template/Index.vm) after creating an account.
The project is OASIS3.


# Citation
This work will be presented at the International Symposium on Biomedical Imaging (ISBI) in April 2020.
An extended version has been published in MDPI Applied sciences.
If you use this package or parts of it, please cite one of the following work:
- [Benchmarking Deep Nets MRI Reconstruction Models on the FastMRI Publicly Available Dataset](https://hal.inria.fr/hal-02436223)
- [Benchmarking MRI Reconstruction Neural Networks on Large Public Datasets](https://www.mdpi.com/2076-3417/10/5/1816)
- [XPDNet for MRI Reconstruction: an Application to the fastMRI 2020 Brain Challenge](https://arxiv.org/abs/2010.07290)
- Density Compensated Unrolled Networks for Non-Cartesian MRI Reconstruction, submitted to ISBI 2021
