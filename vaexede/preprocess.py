import numpy as np

    # to deal with the pre-saved models 
try:
    import importlib.resources as pkg_resources
except ImportError:
    # try backported to python <3.7 `importlib_resources`.
    import importlib_resources as pkg_resources
from . import trained_models  # relative-import the *package* containing the templates

lmin = 30
def preprocess(x, model):
    """`model` must be a string with either `lcdm` or `ede`.
    """
    if model == 'lcdm':
        mmean = np.load(pkg_resources.open_binary(trained_models, 'log10_ratio_mean_new_lcdm.npy'))
        sstd = np.load(pkg_resources.open_binary(trained_models, 'log10_ratio_std_new_lcdm.npy'))
        ref_spectrum = np.load(pkg_resources.open_binary(trained_models, 'DlTT_LCDM_ref.npy'))[lmin:]
    elif model == 'ede':
        mmean = np.load(pkg_resources.open_binary(trained_models, 'log10_ratio_mean_new_ede.npy'))
        sstd = np.load(pkg_resources.open_binary(trained_models, 'log10_ratio_std_new_ede.npy'))
        ref_spectrum = np.load(pkg_resources.open_binary(trained_models, 'DlTT_EDE_ref.npy'))[lmin:]
    x_ = x/ref_spectrum
    x_ = np.log10(x_)
    x_ = (x_-mmean)/sstd
    return x_
    
def unpreprocess(x, model):
    """`model` must be a string with either `lcdm` or `ede`.
    """
    if model not in ['lcdm', 'ede']:
        raise ValueError("To load a model, indicate either 'lcdm' or 'ede'.")
    if model == 'lcdm':
        mmean = np.load(pkg_resources.open_binary(trained_models, 'log10_ratio_mean_new_lcdm.npy'))
        sstd = np.load(pkg_resources.open_binary(trained_models, 'log10_ratio_std_new_lcdm.npy'))
        ref_spectrum = np.load(pkg_resources.open_binary(trained_models, 'DlTT_LCDM_ref.npy'))[lmin:]
    elif model == 'ede':
        mmean = np.load(pkg_resources.open_binary(trained_models, 'log10_ratio_mean_new_ede.npy'))
        sstd = np.load(pkg_resources.open_binary(trained_models, 'log10_ratio_std_new_ede.npy'))
        ref_spectrum = np.load(pkg_resources.open_binary(trained_models, 'DlTT_EDE_ref.npy'))[lmin:]   
    x_ = x*sstd + mmean
    x_ = 10**x_
    x_ *= ref_spectrum
    return x_