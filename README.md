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

All the neural networks are implemented in TensorFlow with the Keras API.


## How to train the neural networks
The scripts to train the neural networks are located in `fastmri_recon/training_scripts/`.
You just need to install the package and its dependencies:
```
pip install . &&\
pip install -r requirements.txt
```


## How to write a new neural network for reconstruction
A good example of a simple neural network on which you can improve is the `zerofill_net` which is simply performing zero-filled reconstruction using `keras`.
The building blocks can then be found in `fastmri_recon/models/utils/`


# Citation
This work will be presented at the International Symposium on Biomedical Imaging (ISBI) in April 2020.
If you use this package or parts of it, please cite the following work: [Benchmarking Deep Nets MRI Reconstruction Models on the FastMRI Publicly Available Dataset](https://hal.inria.fr/hal-02436223)
