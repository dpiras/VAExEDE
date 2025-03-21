# VAExEDE

![](https://img.shields.io/badge/Python-181717?style=plastic&logo=python)
![](https://img.shields.io/badge/Author-Davide%20Piras%20-181717?style=plastic)
![](https://img.shields.io/badge/Installation-pip%20install%20vaexede-181717?style=plastic)
[![arXiv](https://img.shields.io/badge/arXiv-2502.09810-b31b1b.svg)](https://arxiv.org/abs/2502.09810)

A repository to host the trained models from [https://arxiv.org/abs/2502.09810](https://arxiv.org/abs/2502.09810v1), where we trained a variational autoencoder (VAE) on CMB temperature power spectra for early dark energy (EDE) models, to discover novel, data-driven parametrizations.  

## Installation

To use the trained models, follow these steps:
1. (optional) `conda create -n vaexede python=3.11 jupyter` (create a custom `conda` environment with python 3.11) 
2. (optional) `conda activate vaexede` (activate it)
3. Install the package:

        pip install vaexede
        python3 -c 'from vaexede.load_models import load_model'

   or alternatively, clone the repository and install it:

        git clone https://github.com/dpiras/VAExEDE.git
        cd VAExEDE
        pip install . 
        python3 -c 'from vaexede.load_models import load_model'

The latter option will also give you access to a [Jupyter notebook with a quick walkthrough](https://github.com/dpiras/VAExEDE/blob/main/notebooks/quickstart.ipynb) on how to use the models.

## Usage

A simple way to load and use the trained models looks like this:

    model = 'lcdm' # either 'lcdm' or 'ede'
    lcdm_model = load_model(model) # load the trained network

    # add your unitless D_ell temperature spectrum here, in the ell range [30, 2500]
    # should also support batches of data, has not been tested though
    # D_ell = C_ell * ell * (ell+1) / 2pi
    input_spectrum = # your spectrum here

    # and preprocess it as described in the paper
    input_spectrum_preprocess = preprocess(input_spectrum, model)

    # here the preprocessed spectrum gets encoded, samples from the latent space are obtained
    # then the spectrum is decoded and unpreprocessed
    mean, logvar = lcdm_model.encode(input_spectrum_preprocess.reshape(1, -1))
    z = lcdm_model.reparameterize(mean, logvar) # here we sample from the latent distribution
    decoded_spectrum = lcdm_model.decode(z)
    output_spectrum_lcdm = unpreprocess(decoded_spectrum[0, :, 0], model)

## Disclaimer

The repository contains only some of the material needed to reproduce the paper. If you need more or would like to add a feature, feel free to [fork](https://github.com/dpiras/VAExEDE/fork) this repository to work on it; otherwise, please [raise an issue](https://github.com/dpiras/VAExEDE/issues) or contact [Davide Piras](mailto:dr.davide.piras@gmail.com).

## Contributors

[Laura Herold](https://github.com/LauraHerold) and [Luisa Lucie-Smith](https://github.com/lluciesmith) contributed to this code.

## Citation

If you use this code, please cite the corresponding paper:

     @article{Piras:2025eip,
     author = "Piras, Davide and Herold, Laura and Lucie-Smith, Luisa and Komatsu, Eiichiro",
     title = "{$\Lambda$CDM and early dark energy in latent space: a data-driven parametrization of the CMB temperature power spectrum}",
     eprint = "2502.09810",
     archivePrefix = "arXiv",
     primaryClass = "astro-ph.CO",
     month = "2",
     year = "2025"
     }


## License

This code is released under the GPL-3 license - see [LICENSE](https://github.com/dpiras/VAExEDE/blob/main/LICENSE.txt)-, subject to the non-commercial use condition - see [LICENSE_EXT](https://github.com/dpiras/VAExEDE/blob/main/LICENSE_EXT.txt).

     VAExEDE
     Copyright (C) 2025 Davide Piras & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

