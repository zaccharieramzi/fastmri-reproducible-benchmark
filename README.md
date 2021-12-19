# fastMRI reproducible benchmark

[![GitHub Workflow Build Status](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/workflows/Continuous%20testing/badge.svg)](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/actions/workflows/test.yml?query=branch%3Amaster)
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
- [NCPDNet](https://arxiv.org/abs/2101.01570), an unrolled reconstruction algorithm for non-cartesian acquisitions, with density-compensation.

All the neural networks are implemented in TensorFlow with the Keras API.
The older ones (don't judge this was the beginning of my thesis) are coded using the functional API.
The more recent ones are coded in the subclassed API and are more modular.
For the LORAKS reconstruction, you will not be able to reconstruct the proper fastMRI data because of the A/D oversampling.

## Reconstruction settings

The main reconstruction settings are cartesian single-coil and multi-coil 2D reconstruction with random and "periodic sampling".
These settings are covered by almost all the networks in this repo, mainly because they are the settings of the fastMRI challenge.

### Other reconstruction settings

__Non-cartesian__: you can reconstruct non-cartesian data using the [NCPDNet](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/blob/master/fastmri_recon/models/subclassed_models/ncpdnet.py).
It relies on the TensorFlow implementation of the NUFFT, [`tfkbnufft`](https://github.com/zaccharieramzi/tfkbnufft).
This network will allow you to work on 2D single-coil and multi-coil data, as well as 3D single-coil data.

__Deep Image Prior__: you can also reconstruct non-cartesian data in an untrained fashion using the [DIP model](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/blob/master/fastmri_recon/evaluate/reconstruction/dip_reconstrution.py).
This idea originated from the [Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior) paper, and was later adapted to MRI reconstruction by different works ([Accelerated MRI with untrained Neural networks](https://arxiv.org/abs/2007.02471), [Time-Dependent Deep Image Prior for Dynamic MRI](https://arxiv.org/abs/1910.01684)).
It currently is only used for 2D non-cartesian data (primarily for computation time reasons), but you can extend it easily to 2D cartesian data and 3D (PRs welcome).


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

## Reproducing the results of the paper
To reproduce the results of the paper for the fastMRI dataset, run the following script (here for the PD contrast):
```
python fastmri_recon/evaluate/scripts/paper_eval.py --contrast CORPD_FBK
```

To reproduce the results of the paper for the OASIS dataset, run the following script (here for less samples):
```
python fastmri_recon/evaluate/scripts/paper_eval_oasis.py --n-samples 100
```

Finally to reproduce the figures of the paper, you will need to use the [`qualitative_validation_for_net`](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/blob/master/experiments/qualitative_validation_for_net.ipynb) notebook.

### Downloading the model checkpoints
The model checkpoints are stored in the [HuggingFace Hub](https://huggingface.co/zaccharieramzi).
You can download them using the following script, which will automatically put them in the correct directory (for example here the fastMRI models):
```
python fastmri_recon/evaluate/scripts/download_checkpoints.py
```

# Data requirements

## fastMRI

The fastMRI data must be located in a directory whose path is stored in the `FASTMRI_DATA_DIR` environment variable.
It can be downloaded on [the official website](https://fastmri.med.nyu.edu/) after submitting a request (bottom of the page).

The package currently supports public single coil and multi coil knee data.

## OASIS

The OASIS data must be located in a directory whose path is stored in the `OASIS_DATA_DIR` environment variable.
It can be downloaded on [the XNAT store](https://central.xnat.org/app/template/Index.vm) after creating an account.
You can for example use the ZIP download available on [this page](https://central.xnat.org/app/action/ProjectDownloadAction/project/OASIS3).
The project is OASIS3.
The whole list of sessions that need to be downloaded is available in [OASIS_list.csv](OASIS_list.csv).


# Citation
This work will be presented at the International Symposium on Biomedical Imaging (ISBI) in April 2020.
An extended version has been published in MDPI Applied sciences.
If you use this package or parts of it, please cite one of the following work:
- [Benchmarking Deep Nets MRI Reconstruction Models on the FastMRI Publicly Available Dataset](https://hal.inria.fr/hal-02436223)
- [Benchmarking MRI Reconstruction Neural Networks on Large Public Datasets](https://www.mdpi.com/2076-3417/10/5/1816)
- [XPDNet for MRI Reconstruction: an Application to the fastMRI 2020 Brain Challenge](https://arxiv.org/abs/2010.07290)
- [Density Compensated Unrolled Networks for Non-Cartesian MRI Reconstruction](https://arxiv.org/abs/2101.01570)

## Applications
This package has been used to perform MRI reconstruction in the following projects (in addition to the ones mentioned above):
- [Is good old GRAPPA dead?](https://arxiv.org/abs/2106.00753)
- [Learning the sampling density in 2D SPARKLING MRI acquisition for optimized image reconstruction](https://arxiv.org/abs/2103.03559)
- [Denoising Score-Matching for Uncertainty Quantification in Inverse Problems](https://arxiv.org/abs/2011.08698)
