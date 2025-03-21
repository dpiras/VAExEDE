from vaexede.models import CVAE

def load_model(model):
    """ A snippet to load one of the two models. 
        `model` must be a string with either `lcdm` or `ede`.
    """
    if model not in ['lcdm', 'ede']:
        raise ValueError("To load a model, indicate either 'lcdm' or 'ede'.")
        
    lmin = 30
    input_dim = 2501 - lmin

    if model == 'lcdm':
        l = 5 # latent space dimensionality
        beta = 1e-5 # beta value
        model_path = f'./trained_models/best_model_{l}dim_{beta}beta_lcdmrefsuffix_30lmin_normlog'
    else:
        l = 8 # latent space dimensionality
        beta = 2e-5 # beta value
        model_path = f'./trained_models/best_model_{l}dim_{beta}beta_ederefsuffix_30lmin_normlog'

    best_model = CVAE(input_dim, l, concat=False) # concatenation can be ignored here
    best_model.load_weights(model_path)
    
    return best_model